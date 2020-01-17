import contextlib
from enum import Enum, unique
from typing import Dict, Iterable

import numpy

from mlpiot.base.action_executor import \
    ActionExecutor, ActionExecutorLifecycleManager
from mlpiot.base.event_extractor import \
    EventExtractor, EventExtractorLifecycleManager
from mlpiot.base.scene_descriptor import \
    SceneDescriptor, SceneDescriptorLifecycleManager
from mlpiot.proto.image_pb2 import Image
from mlpiot.proto.vision_pipeline_management_pb2 import (
    VisionPipelineManagerMetadata, VisionPipelineOverview
)
from .internal.timestamp_utils import set_now


class VisionPipelineManager(object):
    """`VisionPipelineManager`"""

    @unique
    class _State(Enum):
        NOT_INITIALIZED = 0
        INITIALIZED = 1
        PREPARED_FOR_RUNNING_PIPELINE = 2
        ENTERED_FOR_RUNNING_PIPELINE = 3
        RELEASED = 99

    def __init__(self,
                 scene_descriptor: SceneDescriptor,
                 event_extractor: EventExtractor,
                 action_executors: Iterable[ActionExecutor]):
        self.managed_scene_descriptor = \
            SceneDescriptorLifecycleManager(scene_descriptor)

        self.managed_event_extractor = \
            EventExtractorLifecycleManager(event_extractor)

        action_executors = action_executors or ()
        _managed_action_executors = []
        for action_executor in action_executors:
            _managed_action_executors.append(
                ActionExecutorLifecycleManager(action_executor))
        self.managed_action_executors = tuple(_managed_action_executors)

        self._metadata = VisionPipelineManagerMetadata()
        self._state = VisionPipelineManager._State.NOT_INITIALIZED

    def initialize(
            self,
            environ: Dict[str, str],
            pipeline_manager_metadata: VisionPipelineManagerMetadata):

        assert self._state is VisionPipelineManager._State.NOT_INITIALIZED
        assert \
            isinstance(pipeline_manager_metadata,
                       VisionPipelineManagerMetadata), \
            f"{pipeline_manager_metadata} is not an instance of' \
            ' VisionPipelineManagerMetadata"

        self.managed_scene_descriptor.initialize(environ)
        self.managed_event_extractor.initialize(environ)
        for managed_action_executor in self.managed_action_executors:
            managed_action_executor.initialize(environ)

        self._metadata = pipeline_manager_metadata
        self._state = VisionPipelineManager._State.INITIALIZED

    class _PreparedForRunningPipeline(contextlib.ExitStack):
        def __init__(
                self, pipeline_manager: 'VisionPipelineManager'):

            super().__init__()

            self.pipeline_manager = pipeline_manager

            self.prepared_scene_descriptor = \
                pipeline_manager.managed_scene_descriptor. \
                prepare_for_describing()
            self.enter_context(self.prepared_scene_descriptor)

            self.prepared_event_extractor = \
                pipeline_manager.managed_event_extractor. \
                prepare_for_event_extraction()
            self.enter_context(self.prepared_event_extractor)

            _prepared_action_executors = []
            for managed_action_executor in \
                    pipeline_manager.managed_action_executors:
                prepared_action_executor = \
                    managed_action_executor.\
                    prepare_for_action_execution()
                self.enter_context(prepared_action_executor)
                _prepared_action_executors.append(prepared_action_executor)
            self.prepared_action_executors = tuple(_prepared_action_executors)

        def __enter__(self):
            assert self.pipeline_manager._state is \
                VisionPipelineManager._State.PREPARED_FOR_RUNNING_PIPELINE
            super().__enter__()
            self.pipeline_manager._state = \
                VisionPipelineManager._State.ENTERED_FOR_RUNNING_PIPELINE
            return self

        def __exit__(self, type_, value, traceback):
            assert self.pipeline_manager._state is \
                VisionPipelineManager._State.ENTERED_FOR_RUNNING_PIPELINE
            suppress_exception = super().__exit__(type_, value, traceback)
            self.pipeline_manager._state = \
                VisionPipelineManager._State.RELEASED
            return suppress_exception

        def run_pipeline(
                self,
                input_np_image: numpy.ndarray,
                input_proto_image: Image,
                output_vision_pipeline_overview: VisionPipelineOverview):

            output = output_vision_pipeline_overview

            assert self.pipeline_manager._state is \
                VisionPipelineManager._State.ENTERED_FOR_RUNNING_PIPELINE
            assert \
                isinstance(input_np_image, numpy.ndarray), \
                f"given {input_np_image} is not an instance of numpy.ndarray"
            assert \
                isinstance(input_proto_image, Image), \
                f"given {input_proto_image} is not an instance of Image"
            assert \
                isinstance(output, VisionPipelineOverview), \
                f"given {output} is not an instance of VisionPipelineOverview"

            output.cycle_id = input_proto_image.cycle_id
            set_now(output.timestamp)
            output.metadata.CopyFrom(self.pipeline_manager._metadata)
            output.input_image.CopyFrom(input_proto_image)

            output.scene_description.cycle_id = input_proto_image.cycle_id
            self.prepared_scene_descriptor.describe_scene(
                input_np_image, input_proto_image, output.scene_description)

            output.extracted_events.cycle_id = input_proto_image.cycle_id
            self.prepared_event_extractor.extract_events(
                output.scene_description,
                output.extracted_events)

            for prepared_action_executor in self.prepared_action_executors:
                action_execution = output.action_executions.add()
                prepared_action_executor.execute_action(
                    output.extracted_events, action_execution)
                action_execution.cycle_id = input_proto_image.cycle_id

    def prepare_for_running_pipeline(self):
        assert self._state is \
            VisionPipelineManager._State.INITIALIZED
        prepared = \
            VisionPipelineManager._PreparedForRunningPipeline(self)
        self._state = \
            VisionPipelineManager._State.PREPARED_FOR_RUNNING_PIPELINE
        return prepared

import contextlib
import sys
from typing import Dict, Iterable

import numpy

from mlpiot.base.action_executor import ActionExecutor
from mlpiot.base.event_extractor import EventExtractor
from mlpiot.base.scene_descriptor import SceneDescriptor
from mlpiot.proto.image_pb2 import Image
from mlpiot.proto.vision_pipeline_management_pb2 import (
    VisionPipelineManagerMetadata, VisionPipelineOverview
)
from .internal.timestamp_utils import set_now


class VisionPipelineManager(contextlib.AbstractContextManager):
    """`VisionPipelineManager`"""

    __NOT_INITIALIZED = 0
    __INITIALIZED = 1
    __PREPARED = 2

    def __init__(self,
                 scene_descriptor: SceneDescriptor,
                 event_extractor: EventExtractor,
                 action_executors: Iterable[ActionExecutor]):
        assert \
            isinstance(scene_descriptor, SceneDescriptor), \
            f"{scene_descriptor} is not an instance of SceneDescriptor"
        self.scene_descriptor = scene_descriptor

        assert \
            isinstance(event_extractor, EventExtractor), \
            f"{event_extractor} is not an instance of EventExtractor"
        self.event_extractor = event_extractor

        action_executors = action_executors or ()
        for action_executor in action_executors:
            assert \
                isinstance(action_executor, ActionExecutor), \
                f"{action_executor} is not an instance of ActionExecutor"
        self.action_executors = tuple(action_executors)

        self._metadata = VisionPipelineManagerMetadata()
        self._state = VisionPipelineManager.__NOT_INITIALIZED

    def initialize(
            self,
            environ: Dict[str, str],
            pipeline_manager_metadata: VisionPipelineManagerMetadata):

        assert self._state == VisionPipelineManager.__NOT_INITIALIZED
        assert \
            isinstance(pipeline_manager_metadata,
                       VisionPipelineManagerMetadata), \
            f"{pipeline_manager_metadata} is not an instance of' \
            ' VisionPipelineManagerMetadata"

        self.scene_descriptor.initialize(environ)
        self.event_extractor.initialize(environ)
        for action_executor in self.action_executors:
            action_executor.initialize(environ)

        self._metadata = pipeline_manager_metadata
        self._state = VisionPipelineManager.__INITIALIZED

    def __enter__(self):
        assert self._state == VisionPipelineManager.__INITIALIZED

        for contextmanager in (
                self.scene_descriptor, self.event_extractor,
                *self.action_executors):
            try:
                contextmanager.__enter__()
            except Exception:
                self.__exit__(*sys.exc_info())

        self._state = VisionPipelineManager.__PREPARED

    def __exit__(self, type, value, traceback):
        suppress_exception = False
        for contextmanager in (
                self.scene_descriptor, self.event_extractor,
                *self.action_executors):
            try:
                if contextmanager.is_prepared():
                    suppress_exception = contextmanager.__exit__(
                        type, value, traceback) or suppress_exception
            except Exception:
                pass
        return suppress_exception

    def run_pipeline(self,
                     input_np_image: numpy.ndarray,
                     input_proto_image: Image,
                     output_vision_pipeline_overview: VisionPipelineOverview):

        output = output_vision_pipeline_overview

        assert self._state == VisionPipelineManager.__PREPARED
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
        output.metadata.CopyFrom(self._metadata)
        output.input_image.CopyFrom(input_proto_image)

        output.scene_description.cycle_id = input_proto_image.cycle_id
        self.scene_descriptor.describe_scene(
            input_np_image, input_proto_image, output.scene_description)

        output.extracted_events.cycle_id = input_proto_image.cycle_id
        self.event_extractor.extract_events(
            output.scene_description,
            output.extracted_events)

        for i in range(len(self.action_executors)):
            action_execution = output.action_executions.add()
            action_executor = self.action_executors[i]
            action_executor.execute_action(
                output.extracted_events, action_execution)
            action_execution.cycle_id = input_proto_image.cycle_id

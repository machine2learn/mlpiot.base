import contextlib
from enum import Enum, unique
from typing import Dict, Iterable, Optional

from mlpiot.base.action_executor import \
    ActionExecutor, ActionExecutorLifecycleManager
from mlpiot.base.event_extractor import \
    EventExtractor, EventExtractorLifecycleManager
from mlpiot.base.scene_descriptor import \
    SceneDescriptor, SceneDescriptorLifecycleManager
from mlpiot.base.trainer import \
    Trainer, TrainerLifecycleManager
from mlpiot.base.utils.timestamp import set_now
from mlpiot.proto import \
    ImageWithHelpers, VisionPipelineData, VisionPipelineManagerMetadata


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
                 action_executors: Iterable[ActionExecutor] = (),
                 trainer: Optional[Trainer] = None):
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

        self.trainer = \
            TrainerLifecycleManager(trainer) if trainer is not None else None

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
                input_image: ImageWithHelpers,
                vision_pipeline_data: VisionPipelineData):
            """TODO: fields of vision_pipeline_data should be already filled"""

            assert self.pipeline_manager._state is \
                VisionPipelineManager._State.ENTERED_FOR_RUNNING_PIPELINE
            assert isinstance(vision_pipeline_data, VisionPipelineData)

            set_now(vision_pipeline_data.timestamp)
            vision_pipeline_data.metadata.CopyFrom(
                self.pipeline_manager._metadata)

            self.prepared_scene_descriptor.describe_scene(
                input_image,
                vision_pipeline_data.scene_description)
            self.prepared_event_extractor.extract_events(
                vision_pipeline_data.scene_description,
                vision_pipeline_data.event_extraction)
            for prepared_action_executor in self.prepared_action_executors:
                action_execution = vision_pipeline_data.action_executions.add()
                prepared_action_executor.execute_action(
                    vision_pipeline_data.event_extraction,
                    action_execution)

    def prepare_for_running_pipeline(self):
        assert self._state is \
            VisionPipelineManager._State.INITIALIZED
        prepared = \
            VisionPipelineManager._PreparedForRunningPipeline(self)
        self._state = \
            VisionPipelineManager._State.PREPARED_FOR_RUNNING_PIPELINE
        return prepared

import calendar
from datetime import datetime

import numpy

from mlpiot.base.action_executor import ActionExecutor
from mlpiot.base.event_extractor import EventExtractor
from mlpiot.base.scene_descriptor import SceneDescriptor
from mlpiot.proto.google_timestamp_pb2 import Timestamp
from mlpiot.proto.image_pb2 import Image
from mlpiot.proto.action_execution_pb2 import (
    ActionExecution, ActionExecutorMetadata
)
from mlpiot.proto.event_extraction_pb2 import (
    ExtractedEvents, EventExtractorMetadata
)
from mlpiot.proto.pipeline_overview_pb2 import (
    VisionPipelineOverview, VisionPipelineManagerMetadata
)
from mlpiot.proto.scene_description_pb2 import (
    SceneDescription, SceneDescriptorMetadata
)

_NANOS_PER_MICROSECOND = 1000


def set_now(timestamp):
    dt = datetime.utcnow()
    timestamp.seconds = calendar.timegm(dt.utctimetuple())
    timestamp.nanos = dt.microsecond * _NANOS_PER_MICROSECOND


class VisionPipelineManager(object):
    """`VisionPipelineManager`"""

    __INITIALIZED = 0
    __METADATA_IS_SET = 1
    __PREPARED = 2

    def __init__(self,
                 scene_descriptor, event_extractor, action_executors):
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

        self._metadata = None
        self._state = None


    def initialize(self, environ):
        self.scene_descriptor.lifecycle_initialize(environ)
        self.event_extractor.lifecycle_initialize(environ)
        for action_executor in self.action_executors:
            action_executor.lifecycle_initialize(environ)

        self._state = VisionPipelineManager.__INITIALIZED

    def set_metadata(self, pipeline_manager_metadata):
        assert self._state == VisionPipelineManager.__INITIALIZED, \
            "initialize() should be called before set_metadata()"
        assert \
            isinstance(pipeline_manager_metadata,
                       VisionPipelineManagerMetadata), \
            f"{pipeline_manager_metadata} is not an instance of VisionPipelineManagerMetadata"
        self._metadata = pipeline_manager_metadata
        self._state = VisionPipelineManager.__METADATA_IS_SET

    def prepare(self):
        assert self._state == VisionPipelineManager.__METADATA_IS_SET, \
            "set_metadata() should be called before prepare()"

        self.scene_descriptor_metadata = self.scene_descriptor.lifecycle_prepare()
        self.event_extractor_metadata = self.event_extractor.lifecycle_prepare()
        self.action_executors_metadata = []
        for action_executor in self.action_executors:
            md = action_executor.lifecycle_prepare()
            self.action_executors_metadata.append(md)
        self.action_executors_metadata = tuple(self.action_executors_metadata)

        self._state = VisionPipelineManager.__PREPARED

    def run_pipeline(self,
                     input_np_image, input_proto_image,
                     output_pipeline_overview):
        assert self._state == VisionPipelineManager.__PREPARED, \
            "prepare() should be called before run_pipeline()"
        assert \
            isinstance(input_np_image, numpy.ndarray), \
            f"given {input_np_image} is not an instance of numpy.ndarray"
        assert \
            isinstance(input_proto_image, Image), \
            f"given {input_proto_image} is not an instance of Image"
        assert \
            isinstance(output_pipeline_overview, VisionPipelineOverview), \
            f"given {output_pipeline_overview} is not an instance of VisionPipelineOverview"

        output_pipeline_overview.cycle_id = input_proto_image.cycle_id
        set_now(output_pipeline_overview.timestamp)
        output_pipeline_overview.metadata.CopyFrom(self._metadata)
        output_pipeline_overview.input_image.CopyFrom(input_proto_image)

        self.scene_descriptor.lifecycle_set_input_size(
            input_proto_image.height,
            input_proto_image.width,
            input_proto_image.channels)

        self.scene_descriptor.lifecycle_describe_scene(input_np_image, output_pipeline_overview.scene_description)
        output_pipeline_overview.scene_description.cycle_id = input_proto_image.cycle_id
        set_now(output_pipeline_overview.scene_description.timestamp)
        output_pipeline_overview.scene_description.metadata.CopyFrom(self.scene_descriptor_metadata)

        self.event_extractor.lifecycle_extract_events(
            output_pipeline_overview.scene_description, output_pipeline_overview.extracted_events)
        output_pipeline_overview.extracted_events.cycle_id = input_proto_image.cycle_id
        set_now(output_pipeline_overview.extracted_events.timestamp)
        output_pipeline_overview.extracted_events.metadata.CopyFrom(self.event_extractor_metadata)

        for i in range(len(self.action_executors)):
            action_execution = output_pipeline_overview.action_executions.add()
            action_executor = self.action_executors[i]
            action_executor.lifecycle_execute_action(
                output_pipeline_overview.extracted_events, action_execution)
            action_execution.cycle_id = input_proto_image.cycle_id
            set_now(action_execution.timestamp)
            action_execution.metadata.CopyFrom(self.action_executors_metadata[i])

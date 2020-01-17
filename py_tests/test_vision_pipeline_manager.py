"""Tests for mlpiot.base.vision_pipeline_manager"""

import unittest

import numpy as np

from mlpiot.base.action_executor import ActionExecutor
from mlpiot.base.event_extractor import EventExtractor
from mlpiot.base.scene_descriptor import SceneDescriptor
from mlpiot.base.vision_pipeline_manager import VisionPipelineManager
from mlpiot.proto.image_pb2 import Image
from mlpiot.proto.vision_pipeline_management_pb2 import \
    VisionPipelineManagerMetadata, VisionPipelineOverview


class DummySceneDescriptor(SceneDescriptor):
    def initialize(self, environ):
        pass

    def prepare_for_describing(self, output_metadata):
        pass

    def describe_scene(
            self, input_np_image, input_proto_image, output_scene_description):
        pass


class DummyEventExtractor(EventExtractor):
    def initialize(self, environ):
        pass

    def prepare_for_event_extraction(self, output_metadata):
        pass

    def extract_events(
            self, input_scene_description, output_extracted_events):
        pass


class DummyActionExecutor(ActionExecutor):
    def initialize(self, environ):
        pass

    def prepare_for_action_execution(self, output_metadata):
        pass

    def execute_action(
            self, input_extracted_events, output_action_execution):
        pass


class TestVisionPipelineManager(unittest.TestCase):
    """Test mlpiot.base.vision_pipeline_manager.VisionPipelineManager"""

    def test_smoke(self):
        "A simple test to check if everything is importable"

        dummy_scene_descriptor = DummySceneDescriptor()
        dummy_event_extractor = DummyEventExtractor()
        dummy_action_executor = DummyActionExecutor()

        vision_pipeline_manager = VisionPipelineManager(
            dummy_scene_descriptor, dummy_event_extractor,
            [dummy_action_executor])

        vpmm = VisionPipelineManagerMetadata()
        vision_pipeline_manager.initialize(
            {}, vpmm)

        with vision_pipeline_manager.\
                prepare_for_running_pipeline() as pipeline_runner:
            input_np_image = np.array([[[1]]])
            input_proto_image = Image()
            input_proto_image.cycle_id = 1001
            input_proto_image.height = 1
            input_proto_image.width = 1
            input_proto_image.channels = 1
            output_pipeline_overview = VisionPipelineOverview()
            pipeline_runner.run_pipeline(
                input_np_image, input_proto_image, output_pipeline_overview)

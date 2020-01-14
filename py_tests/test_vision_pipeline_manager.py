"""Tests for mlpiot.base.vision_pipeline_manager"""

import unittest

import numpy as np

from mlpiot.base.action_executor import (
    ActionExecutor, ActionExecutorMetadata
)
from mlpiot.base.event_extractor import (
    EventExtractor, EventExtractorMetadata
)
from mlpiot.base.scene_descriptor import (
    SceneDescriptor, SceneDescriptorMetadata
)
from mlpiot.base.vision_pipeline_manager import (
    VisionPipelineManager, VisionPipelineManagerMetadata,
    VisionPipelineOverview
)
from mlpiot.proto.image_pb2 import Image


class DummySceneDescriptor(SceneDescriptor):
    def initialize(self, environ):
        pass

    def prepare(self):
        return SceneDescriptorMetadata()

    def describe_scene_impl(
            self, input_np_image, input_proto_image, output_scene_description):
        pass


class DummyEventExtractor(EventExtractor):
    def initialize(self, environ):
        pass

    def prepare(self):
        return EventExtractorMetadata()

    def extract_events_impl(
            self, input_scene_description, output_extracted_events):
        pass


class DummyActionExecutor(ActionExecutor):
    def initialize(self, environ):
        pass

    def prepare(self):
        return ActionExecutorMetadata()

    def execute_action_impl(
            self, input_extracted_events, output_action_execution):
        pass


class TestVisionPipelineManager(unittest.TestCase):
    """Test mlpiot.base.vision_pipeline_manager.VisionPipelineManager"""

    def test_smoke(self):
        "A simple test to check if everything is importable"

        envs = {}

        dummy_scene_descriptor = DummySceneDescriptor(envs)
        dummy_event_extractor = DummyEventExtractor(envs)
        dummy_action_executor = DummyActionExecutor(envs)

        vision_pipeline_manager = VisionPipelineManager(
            dummy_scene_descriptor, dummy_event_extractor,
            [dummy_action_executor])

        vpmm = VisionPipelineManagerMetadata()
        vision_pipeline_manager.set_metadata(vpmm)

        with vision_pipeline_manager:
            input_np_image = np.array([[[1]]])
            input_proto_image = Image()
            input_proto_image.cycle_id = 1001
            input_proto_image.height = 1
            input_proto_image.width = 1
            input_proto_image.channels = 1
            output_pipeline_overview = VisionPipelineOverview()
            vision_pipeline_manager.run_pipeline(
                input_np_image, input_proto_image, output_pipeline_overview)

"""Tests for mlpiot.base.vision_pipeline_manager"""

import unittest

from mlpiot.base.action_executor import ActionExecutor
from mlpiot.base.event_extractor import EventExtractor
from mlpiot.base.scene_descriptor import SceneDescriptor
from mlpiot.base.trainer import Trainer
from mlpiot.base.vision_pipeline_manager import VisionPipelineManager
from mlpiot.proto import \
    Image, ImageWithHelpers, \
    VisionPipelineData, VisionPipelineManagerMetadata


class DummySceneDescriptor(SceneDescriptor):
    def initialize(self, environ):
        pass

    def prepare_for_describing(self, output_metadata):
        pass

    def describe_scene(self, input_image, output_scene_description):
        pass


class DummyEventExtractor(EventExtractor):
    def initialize(self, environ):
        pass

    def prepare_for_event_extraction(self, output_metadata):
        pass

    def extract_events(
            self, input_scene_description, output_event_extraction):
        pass


class DummyActionExecutor(ActionExecutor):
    def initialize(self, environ):
        pass

    def prepare_for_action_execution(self, output_metadata):
        pass

    def execute_action(
            self, input_event_extraction, output_action_execution):
        pass


class DummyTrainer(Trainer):
    def initialize(self, environ):
        pass

    def prepare_for_training(self, output_metadata):
        pass

    def train(self, dataset):
        pass


class TestVisionPipelineManager(unittest.TestCase):
    """Test mlpiot.base.vision_pipeline_manager.VisionPipelineManager"""

    def test_smoke(self):
        "A simple test to check if everything is importable"

        dummy_scene_descriptor = DummySceneDescriptor()
        dummy_event_extractor = DummyEventExtractor()
        dummy_action_executor = DummyActionExecutor()
        dummy_trainer = DummyTrainer()

        vision_pipeline_manager = VisionPipelineManager(
            dummy_scene_descriptor,
            dummy_event_extractor,
            [dummy_action_executor],
            dummy_trainer)

        vpmm = VisionPipelineManagerMetadata()
        vision_pipeline_manager.initialize({}, vpmm)

        with vision_pipeline_manager.\
                prepare_for_running_pipeline() as pipeline_runner:

            input_image_proto = Image()
            input_image_proto.height = 1
            input_image_proto.width = 1
            input_image_proto.channels = 1
            input_image = ImageWithHelpers(input_image_proto)

            vision_pipeline_data = VisionPipelineData()
            vision_pipeline_data.id = 1001

            pipeline_runner.run_pipeline(input_image, vision_pipeline_data)

        initialized_trainer = vision_pipeline_manager.managed_trainer

        with initialized_trainer.prepare_for_training() as ready_runner:
            ready_runner.train([vision_pipeline_data])

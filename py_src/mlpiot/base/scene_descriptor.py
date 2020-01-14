from abc import abstractmethod
import contextlib

import numpy

from mlpiot.proto.image_pb2 import Image
from mlpiot.proto.scene_description_pb2 import \
    SceneDescription, SceneDescriptorMetadata

from .internal.timestamp_utils import set_now


class SceneDescriptor(contextlib.AbstractContextManager):
    """`SceneDescriptor` is a base for vision codes interpreting a given image
    and describing it as a `SceneDescription`.

    The lifecycle of an instance of this class will be managed by a pipeline
    manager which is going to initialize it, feed it, and pass its output to
    an `EventExtractor`."""

    __INITIALIZED = 0
    __PREPARED = 1

    def __init__(self, environ):
        "Called by a lifecycle manager"
        self.initialize(environ)
        self._state = SceneDescriptor.__INITIALIZED
        self._metadata = None

    def __enter__(self):
        "See contextmanager.__enter__()"
        assert self._state == SceneDescriptor.__INITIALIZED
        self._metadata = self.prepare()
        assert \
            isinstance(self._metadata, SceneDescriptorMetadata), \
            f"{self._metadata} returned by prepare is not an instance of" \
            " SceneDescriptorMetadata"
        self._state = SceneDescriptor.__PREPARED
        return self

    def is_prepared(self):
        return self._state == SceneDescriptor.__PREPARED

    def describe_scene(
            self,
            input_np_image: numpy.ndarray,
            input_proto_image: Image,
            output_scene_description: SceneDescription):
        "Called by a lifecycle manager"
        assert self._state == SceneDescriptor.__PREPARED
        self.describe_scene_impl(
            input_np_image, input_proto_image, output_scene_description)
        output_scene_description.metadata.CopyFrom(self._metadata)
        set_now(output_scene_description.timestamp)

    def __exit__(self, type, value, traceback):
        "See contextmanager.__exit__(type, value, traceback)"
        pass

    @abstractmethod
    def initialize(self, environ) -> None:
        """Initializes the `SceneDescriptor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        pass

    @abstractmethod
    def prepare(self) -> SceneDescriptorMetadata:
        """Loads the internal components and return a `SceneDescriptorMetadata`

        Called once after the `SceneDescriptor` is initialized but before
        starting the loop which calls `describe_scene`"""
        pass

    @abstractmethod
    def describe_scene_impl(
            self,
            input_np_image: numpy.ndarray,
            input_proto_image: Image,
            output_scene_description: SceneDescription) -> None:
        """Fills the given `SceneDescription` describing the input.

        input_np_image -- image as a numpy array
        input_proto_image -- image metadata and optionally image content as an
            mlpiot.proto.image_pb2.Image object
        output_scene_description -- which will be passed to an `EventExtractor`
        """
        pass

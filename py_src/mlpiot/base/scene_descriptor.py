from abc import ABC, abstractmethod

from mlpiot.proto.scene_description_pb2 import SceneDescriptorMetadata


class SceneDescriptor(ABC):
    """`SceneDescriptor` is a base for vision codes interpreting a given image
    and describing it as a `SceneDescription`.

    The lifecycle of an instance of this class will be managed by a pipeline
    manager which is going to initialize it, feed it, and pass its output to
    an `EventExtractor`."""

    __INITIALIZED = 0
    __PREPARED = 1
    __INPUT_SIZE_SET = 2

    def __init__(self):
        self._input_height = None
        self._input_width = None
        self._input_channels = None

    def lifecycle_initialize(self, environ):
        """Called by a lifecycle manager"""
        self.initialize(environ)
        self._state = SceneDescriptor.__INITIALIZED

    def lifecycle_prepare(self):
        """Called by a lifecycle manager"""
        assert self._state == SceneDescriptor.__INITIALIZED
        metadata = self.prepare()
        assert \
            isinstance(metadata, SceneDescriptorMetadata), \
            f"{metadata} returned by prepare is not an instance of" \
            " SceneDescriptorMetadata"
        self._state = SceneDescriptor.__PREPARED
        return metadata

    def lifecycle_set_input_size(self, height, width, channels):
        """Called by a lifecycle manager"""
        assert self._state in (
            SceneDescriptor.__PREPARED,
            SceneDescriptor.__INPUT_SIZE_SET)
        if self._state == SceneDescriptor.__INPUT_SIZE_SET and \
                self._input_height == height and \
                self._input_width == width and \
                self._input_channels == channels:
            return
        self._input_height = height
        self._input_width = width
        self._input_channels = channels
        self.on_input_size_update()
        self._state = SceneDescriptor.__INPUT_SIZE_SET

    def lifecycle_describe_scene(
            self, input_np_image, output_scene_description):
        assert self._state == SceneDescriptor.__INPUT_SIZE_SET
        self.describe_scene(input_np_image, output_scene_description)

    @abstractmethod
    def initialize(self, environ):
        """Initializes the `SceneDescriptor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        pass

    @abstractmethod
    def prepare(self):
        """Loads the internal components and return a `SceneDescriptorMetadata`

        Called once after the `SceneDescriptor` is initialized but before
        starting the loop which calls `describe_scene`"""
        pass

    @abstractmethod
    def on_input_size_update(self):
        """Notifies the `SceneDescriptor` about input size being updated.

        This method is called at least once, after the internal variables
        regarding input size is set/updated, before starting the main loop. It
        may be called more than once, for example if the size of an input image
        is going to be different from the previous one."""
        pass

    @abstractmethod
    def describe_scene(self, input_np_image, output_scene_description):
        """Fills the given `SceneDescription` describing the input.

        input_np_image -- a numpy array with shape
               (self._input_height, self._input_width, self._input_channels)
        output_scene_description -- which will be passed to an `EventExtractor`
        """
        pass

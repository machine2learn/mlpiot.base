from abc import ABC, abstractmethod
import contextlib
from enum import Enum, unique
from typing import Dict

from mlpiot.base.utils.timestamp import set_now
from mlpiot.proto import \
    ImageWithHelpers, SceneDescription, SceneDescriptorMetadata


class SceneDescriptor(ABC):
    """`SceneDescriptor` is a base for vision code in a ML pipeline.

    It interprets a given image and describing it as a `SceneDescription`.

    The lifecycle of an instance of this class will be managed by a
    `SceneDescriptorLifecycleManager`"""

    @abstractmethod
    def initialize(self, environ: Dict[str, str]) -> None:
        """Initializes the `SceneDescriptor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_for_describing(
            self,
            output_metadata: SceneDescriptorMetadata):
        """Loads the internal components and fills the given metadata.

        Called once after the `SceneDescriptor` is initialized but before
        starting the loop which calls `describe_scene`

        output_metadata -- to be filled
        """
        raise NotImplementedError

    @abstractmethod
    def describe_scene(
            self,
            input_image: ImageWithHelpers,
            output_scene_description: SceneDescription) -> None:
        """Fills the given `SceneDescription` describing the input."""
        raise NotImplementedError

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        return False


class SceneDescriptorLifecycleManager(object):

    @unique
    class _State(Enum):
        NOT_INITIALIZED = 0
        INITIALIZED = 1
        PREPARED_FOR_DESCRIBING = 2
        ENTERED_FOR_DESCRIBING = 3
        RELEASED = 99

    def __init__(self, implementation: SceneDescriptor):
        assert isinstance(implementation, SceneDescriptor)
        self.implementation = implementation
        self._metadata = SceneDescriptorMetadata()
        self._metadata.name = self.__class__.__name__
        self._state = SceneDescriptorLifecycleManager._State.NOT_INITIALIZED

    def initialize(self, environ: Dict[str, str]) -> None:
        assert self._state is \
            SceneDescriptorLifecycleManager._State.NOT_INITIALIZED
        self.implementation.initialize(environ)
        self._state = SceneDescriptorLifecycleManager._State.INITIALIZED

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        suppress_exception = self.implementation.release(
            type_, value, traceback)
        self._state = SceneDescriptorLifecycleManager._State.RELEASED
        return suppress_exception

    class _PreparedForDesribing(contextlib.AbstractContextManager):
        def __init__(
                self, lifecycle_manager: 'SceneDescriptorLifecycleManager'):
            self.lifecycle_manager = lifecycle_manager

        def __enter__(self):
            assert self.lifecycle_manager._state is \
                SceneDescriptorLifecycleManager._State.PREPARED_FOR_DESCRIBING
            self.lifecycle_manager._state = \
                SceneDescriptorLifecycleManager._State.ENTERED_FOR_DESCRIBING
            return self

        def __exit__(self, type_, value, traceback):
            assert self.lifecycle_manager._state is \
                SceneDescriptorLifecycleManager._State.ENTERED_FOR_DESCRIBING
            return self.lifecycle_manager.release(
                type_, value, traceback)

        def describe_scene(
                self,
                input_image: ImageWithHelpers,
                output_scene_description: SceneDescription):
            assert self.lifecycle_manager._state is \
                SceneDescriptorLifecycleManager._State.ENTERED_FOR_DESCRIBING
            self.lifecycle_manager.implementation.describe_scene(
                input_image, output_scene_description)
            output_scene_description.metadata.CopyFrom(
                self.lifecycle_manager._metadata)
            set_now(output_scene_description.timestamp)

    def prepare_for_describing(self):
        assert self._state is \
            SceneDescriptorLifecycleManager._State.INITIALIZED
        self.implementation.prepare_for_describing(self._metadata)
        prepared = SceneDescriptorLifecycleManager._PreparedForDesribing(self)
        self._state = \
            SceneDescriptorLifecycleManager._State.PREPARED_FOR_DESCRIBING
        return prepared

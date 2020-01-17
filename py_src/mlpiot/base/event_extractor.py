from abc import ABC, abstractmethod
import contextlib
from enum import Enum, unique
from typing import Dict

from mlpiot.proto.event_extraction_pb2 import \
    EventExtractorMetadata, ExtractedEvents
from mlpiot.proto.scene_description_pb2 import SceneDescription
from .internal.timestamp_utils import set_now


class EventExtractor(ABC):
    """`EventExtractor` is a base for a decision maker in a ML pipeline.

    It takes a `SceneDescription` and optionally extracts a number of `Event`s
    from the given description.

    The lifecycle of an instance of this class will be managed by a
    `EventExtractorLifecycleManager`"""

    @abstractmethod
    def initialize(self, environ: Dict[str, str]) -> None:
        """Initializes the `EventExtractor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_for_event_extraction(
            self,
            output_metadata: EventExtractorMetadata):
        """Loads the internal components and return a `EventExtractorMetadata`

        Called once after the `EventExtractor` is initialized but before
        starting the loop which calls `extract_events`

        output_metadata -- to be filled
        """
        raise NotImplementedError

    @abstractmethod
    def extract_events(
            self,
            input_scene_description: SceneDescription,
            output_extracted_events: ExtractedEvents) -> None:
        """Fills the given `ExtractedEvents` putting events extracted from the
        given `scene_description`.

        input_scene_description -- a `SceneDescription` instance
        output_extracted_events -- which will be passed to `ActionExecutor`s
        """
        raise NotImplementedError

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        return False


class EventExtractorLifecycleManager(object):

    @unique
    class _State(Enum):
        NOT_INITIALIZED = 0
        INITIALIZED = 1
        PREPARED_FOR_EXTRACTION = 2
        ENTERED_FOR_EXTRACTION = 3
        RELEASED = 99

    def __init__(self, implementation: EventExtractor):
        assert isinstance(implementation, EventExtractor)
        self.implementation = implementation
        self._metadata = EventExtractorMetadata()
        self._metadata.name = self.__class__.__name__
        self._state = EventExtractorLifecycleManager._State.NOT_INITIALIZED

    def initialize(self, environ) -> None:
        assert self._state is \
            EventExtractorLifecycleManager._State.NOT_INITIALIZED
        self.implementation.initialize(environ)
        self._state = EventExtractorLifecycleManager._State.INITIALIZED

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        suppress_exception = self.implementation.release(
            type_, value, traceback)
        self._state = EventExtractorLifecycleManager._State.RELEASED
        return suppress_exception

    class _PreparedForEventExtraction(contextlib.AbstractContextManager):
        def __init__(
                self, lifecycle_manager: 'EventExtractorLifecycleManager'):
            self.lifecycle_manager = lifecycle_manager

        def __enter__(self):
            assert self.lifecycle_manager._state is \
                EventExtractorLifecycleManager._State.PREPARED_FOR_EXTRACTION
            self.lifecycle_manager._state = \
                EventExtractorLifecycleManager._State.ENTERED_FOR_EXTRACTION
            return self

        def __exit__(self, type_, value, traceback):
            assert self.lifecycle_manager._state is \
                EventExtractorLifecycleManager._State.ENTERED_FOR_EXTRACTION
            return self.lifecycle_manager.release(
                type_, value, traceback)

        def extract_events(
                self,
                input_scene_description: SceneDescription,
                output_extracted_events: ExtractedEvents):
            assert self.lifecycle_manager._state is \
                EventExtractorLifecycleManager._State.ENTERED_FOR_EXTRACTION
            self.lifecycle_manager.implementation.extract_events(
                input_scene_description, output_extracted_events)
            output_extracted_events.metadata.CopyFrom(
                self.lifecycle_manager._metadata)
            set_now(output_extracted_events.timestamp)

    def prepare_for_event_extraction(self):
        assert self._state is \
            EventExtractorLifecycleManager._State.INITIALIZED
        self.implementation.prepare_for_event_extraction(self._metadata)
        prepared = \
            EventExtractorLifecycleManager._PreparedForEventExtraction(self)
        self._state = \
            EventExtractorLifecycleManager._State.PREPARED_FOR_EXTRACTION
        return prepared

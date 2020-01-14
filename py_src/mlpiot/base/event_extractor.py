from abc import abstractmethod
import contextlib

from mlpiot.proto.event_extraction_pb2 import \
    ExtractedEvents, EventExtractorMetadata
from mlpiot.proto.scene_description_pb2 import SceneDescription

from .internal.timestamp_utils import set_now


class EventExtractor(contextlib.AbstractContextManager):
    """`EventExtractor` is a base for a decision maker which takes a
    `SceneDescription` and optionally extracts a number of `Event`s from
    the given description.

    The lifecycle of an instance of this class will be managed by a pipeline
    manager which is going to initialize it, feed it, and pass its output to
    an `ActionExecutor`."""

    __INITIALIZED = 0
    __PREPARED = 1

    def __init__(self, environ):
        "Called by a lifecycle manager"
        self.initialize(environ)
        self._state = EventExtractor.__INITIALIZED
        self._metadata = None

    def __enter__(self):
        "See contextmanager.__enter__()"
        assert self._state == EventExtractor.__INITIALIZED
        self._metadata = self.prepare()
        assert \
            isinstance(self._metadata, EventExtractorMetadata), \
            f"{self._metadata} returned by prepare is not an instance of" \
            " EventExtractorMetadata"
        self._state = EventExtractor.__PREPARED
        return self

    def is_prepared(self):
        return self._state == EventExtractor.__PREPARED

    def extract_events(
            self,
            input_scene_description: SceneDescription,
            output_extracted_events: ExtractedEvents):
        assert self._state == EventExtractor.__PREPARED
        self.extract_events_impl(
            input_scene_description, output_extracted_events)
        output_extracted_events.metadata.CopyFrom(self._metadata)
        set_now(output_extracted_events.timestamp)

    def __exit__(self, type, value, traceback):
        "See contextmanager.__exit__(type, value, traceback)"
        pass

    @abstractmethod
    def initialize(self, params) -> None:
        """Initializes the `EventExtractor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        pass

    @abstractmethod
    def prepare(self) -> EventExtractorMetadata:
        """Loads the internal components and return a `EventExtractorMetadata`

        Called once after the `EventExtractor` is initialized but before
        starting the loop which calls `extract_events`"""
        pass

    @abstractmethod
    def extract_events_impl(
            self,
            input_scene_description: SceneDescription,
            output_extracted_events: ExtractedEvents) -> None:
        """Fills the given `ExtractedEvents` putting events extracted from the
        given `scene_description`.

        input_scene_description -- a `SceneDescription` instance
        output_extracted_events -- which will be passed to `ActionExecutor`s
        """
        pass

from abc import ABC, abstractmethod

from mlpiot.proto.event_extraction_pb2 import EventExtractorMetadata


class EventExtractor(ABC):
    """`EventExtractor` is a base for a decision maker which takes a
    `SceneDescription` and optionally extracts a number of `Event`s from
    the given description.

    The lifecycle of an instance of this class will be managed by a pipeline
    manager which is going to initialize it, feed it, and pass its output to
    an `ActionExecutor`."""

    __INITIALIZED = 0
    __PREPARED = 1

    def __init__(self):
        pass

    def lifecycle_initialize(self, environ):
        """Called by a lifecycle manager"""
        self.initialize(environ)
        self._state = EventExtractor.__INITIALIZED

    def lifecycle_prepare(self):
        """Called by a lifecycle manager"""
        assert self._state == EventExtractor.__INITIALIZED
        metadata = self.prepare()
        assert \
            isinstance(metadata, EventExtractorMetadata), \
            f"{metadata} returned by prepare is not an instance of" \
            " EventExtractorMetadata"
        self._state = EventExtractor.__PREPARED
        return metadata

    def lifecycle_extract_events(
            self, input_scene_description, output_extracted_events):
        assert self._state == EventExtractor.__PREPARED
        self.extract_events(input_scene_description, output_extracted_events)

    @abstractmethod
    def initialize(self, params):
        """Initializes the `EventExtractor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        pass

    @abstractmethod
    def prepare(self):
        """Loads the internal components and return a `EventExtractorMetadata`

        Called once after the `EventExtractor` is initialized but before
        starting the loop which calls `extract_events`"""
        pass

    @abstractmethod
    def extract_events(self, input_scene_description, output_extracted_events):
        """Fills the given `ExtractedEvents` putting events extracted from the
        given `scene_description`.

        input_scene_description -- a `SceneDescription` instance
        output_extracted_events -- which will be passed to `ActionExecutor`s
        """
        pass

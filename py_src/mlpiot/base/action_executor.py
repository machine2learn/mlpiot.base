from abc import ABC, abstractmethod

from mlpiot.proto.action_execution_pb2 import (
    ActionExecution, ActionExecutorMetadata
)


class ActionExecutor(ABC):
    """`ActionExecutor` is a base for the last step in the vision app pipeline,
    which takes an `ExtractedEvents`, and performs some actions in response.

    The lifecycle of an instance of this class will be managed by a pipeline
    manager which is going to initialize it and feed it."""

    __INITIALIZED = 0
    __PREPARED = 1

    def __init__(self):
        pass

    def lifecycle_initialize(self, environ):
        """Called by a lifecycle manager"""
        self.initialize(environ)
        self._state = ActionExecutor.__INITIALIZED

    def lifecycle_prepare(self):
        """Called by a lifecycle manager"""
        assert self._state == ActionExecutor.__INITIALIZED

        metadata = self.prepare()
        assert \
            isinstance(metadata, ActionExecutorMetadata), \
            f"{metadata} returned by prepare is not an instance of ActionExecutorMetadata"
        self._state = ActionExecutor.__PREPARED
        return metadata

    def lifecycle_execute_action(self, input_extracted_events, output_action_execution):
        assert self._state == ActionExecutor.__PREPARED
        self.execute_action(input_extracted_events, output_action_execution)

    @abstractmethod
    def initialize(self, environ):
        """Initializes the `ActionExecutor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        pass

    @abstractmethod
    def prepare(self):
        """Loads the internal components and return a `ActionExecutorMetadata`

        Called once after the `ActionExecutor` is initialized but before
        starting the loop which calls `extract_events`"""
        pass

    @abstractmethod
    def execute_action(self, input_extracted_events, output_action_execution):
        """Performs an action based on the given `extracted_events`. The given
        `ActionExecution` is filled which may contain events emitted through
        performing the actions. These actions are targeted to only be logged
        for debugging and troubleshooting.

        input_extracted_events -- an `ExtractedEvents` instance
        output_action_execution -- logs of the executed action
        """
        pass

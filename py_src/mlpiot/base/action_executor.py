from abc import ABC, abstractmethod
import contextlib
from enum import Enum, unique

from mlpiot.base.utils.timestamp import set_now
from mlpiot.proto import \
    ActionExecution, ActionExecutorMetadata, EventExtraction


class ActionExecutor(ABC):
    """`ActionExecutor` is a base for the last step in a vision app pipeline.

    It takes an `EventExtraction`, and performs some action in response.

    The lifecycle of an instance of this class will be managed by a
    `ActionExecutorLifecycleManager`"""

    @abstractmethod
    def initialize(self, environ) -> None:
        """Initializes the `ActionExecutor` using the given params.

        Validate the given params but if only validating is possible without
        heavy IO operations. Store them to be used in the `prepare` stage.
        Do not load heavy external libraries or models in this stage.

        environ -- a dictionary from parameter name to its string value.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_for_action_execution(
            self,
            output_metadata: ActionExecutorMetadata):
        """Loads the internal components and return a `ActionExecutorMetadata`

        Called once after the `ActionExecutor` is initialized but before
        starting the loop which calls `extract_events`

        output_metadata -- to be filled
        """
        raise NotImplementedError

    @abstractmethod
    def execute_action(
            self,
            input_event_extraction: EventExtraction,
            output_action_execution: ActionExecution) -> None:
        """Performs an action based on the given `event_extraction`. The given
        `ActionExecution` is filled which may contain events emitted through
        performing the actions. These actions are targeted to only be logged
        for debugging and troubleshooting.

        input_event_extraction -- an `EventExtraction` instance
        output_action_execution -- logs of the executed action
        """
        raise NotImplementedError

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        return False


class ActionExecutorLifecycleManager(object):

    @unique
    class _State(Enum):
        NOT_INITIALIZED = 0
        INITIALIZED = 1
        PREPARED_FOR_ACT_EXEC = 2
        ENTERED_FOR_ACT_EXEC = 3
        RELEASED = 99

    def __init__(self, implementation: ActionExecutor):
        assert isinstance(implementation, ActionExecutor)
        self.implementation = implementation
        self._metadata = ActionExecutorMetadata()
        self._metadata.name = self.__class__.__name__
        self._state = ActionExecutorLifecycleManager._State.NOT_INITIALIZED

    def initialize(self, environ) -> None:
        assert self._state is \
            ActionExecutorLifecycleManager._State.NOT_INITIALIZED
        self.implementation.initialize(environ)
        self._state = ActionExecutorLifecycleManager._State.INITIALIZED

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        suppress_exception = self.implementation.release(
            type_, value, traceback)
        self._state = ActionExecutorLifecycleManager._State.RELEASED
        return suppress_exception

    class _PreparedForActionExecution(contextlib.AbstractContextManager):
        def __init__(
                self, lifecycle_manager: 'ActionExecutorLifecycleManager'):
            self.lifecycle_manager = lifecycle_manager

        def __enter__(self):
            assert self.lifecycle_manager._state is \
                ActionExecutorLifecycleManager._State.PREPARED_FOR_ACT_EXEC
            self.lifecycle_manager._state = \
                ActionExecutorLifecycleManager._State.ENTERED_FOR_ACT_EXEC
            return self

        def __exit__(self, type_, value, traceback):
            assert self.lifecycle_manager._state is \
                ActionExecutorLifecycleManager._State.ENTERED_FOR_ACT_EXEC
            return self.lifecycle_manager.release(
                type_, value, traceback)

        def execute_action(
                self,
                input_event_extraction: EventExtraction,
                output_action_execution: ActionExecution):
            assert self.lifecycle_manager._state is \
                ActionExecutorLifecycleManager._State.ENTERED_FOR_ACT_EXEC
            self.lifecycle_manager.implementation.execute_action(
                input_event_extraction, output_action_execution)
            output_action_execution.metadata.CopyFrom(
                self.lifecycle_manager._metadata)
            set_now(output_action_execution.timestamp)

    def prepare_for_action_execution(self):
        assert self._state is \
            ActionExecutorLifecycleManager._State.INITIALIZED
        self.implementation.prepare_for_action_execution(self._metadata)
        prepared = \
            ActionExecutorLifecycleManager._PreparedForActionExecution(self)
        self._state = \
            ActionExecutorLifecycleManager._State.PREPARED_FOR_ACT_EXEC
        return prepared

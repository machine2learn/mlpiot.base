from abc import ABC, abstractmethod
import contextlib
from enum import Enum, unique
from typing import Dict, Optional

from mlpiot.base.utils.dataset import auto_detect_dataset, DatasetParams
from mlpiot.proto import TrainerMetadata, VisionPipelineDataset


class Trainer(ABC):

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
    def prepare_for_training(
            self,
            output_metadata: TrainerMetadata):
        """Loads the internal components and fills the given metadata object"""
        raise NotImplementedError

    @abstractmethod
    def train(
            self,
            dataset: VisionPipelineDataset,
            val_dataset: Optional[VisionPipelineDataset] = None) -> None:
        """TODO"""
        raise NotImplementedError

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        return False

    def get_dataset(
            self,
            directory_path: str,
            dataset_params: DatasetParams) -> VisionPipelineDataset:
        """TODO"""
        return auto_detect_dataset(directory_path, dataset_params)


class TrainerLifecycleManager(object):

    @unique
    class _State(Enum):
        NOT_INITIALIZED = 0
        INITIALIZED = 1
        PREPARED_FOR_TRAINING = 2
        ENTERED_FOR_TRAINING = 3
        RELEASED = 99

    def __init__(self, implementation: Trainer):
        assert isinstance(implementation, Trainer)
        self.implementation = implementation
        self._metadata = TrainerMetadata()
        self._metadata.name = self.__class__.__name__
        self._state = TrainerLifecycleManager._State.NOT_INITIALIZED

    def initialize(self, environ: Dict[str, str]) -> None:
        assert self._state is \
            TrainerLifecycleManager._State.NOT_INITIALIZED
        self.implementation.initialize(environ)
        self._state = TrainerLifecycleManager._State.INITIALIZED

    def release(self, type_, value, traceback) -> bool:
        """release the resources"""
        suppress_exception = self.implementation.release(
            type_, value, traceback)
        self._state = TrainerLifecycleManager._State.RELEASED
        return suppress_exception

    def get_dataset(
            self,
            directory_path: str,
            dataset_params: DatasetParams) -> VisionPipelineDataset:
        return self.implementation.get_dataset(
            directory_path, dataset_params)

    class _PreparedForTraining(contextlib.AbstractContextManager):
        def __init__(
                self, lifecycle_manager: 'TrainerLifecycleManager'):
            self.lifecycle_manager = lifecycle_manager

        def __enter__(self):
            assert self.lifecycle_manager._state is \
                TrainerLifecycleManager._State.PREPARED_FOR_TRAINING
            self.lifecycle_manager._state = \
                TrainerLifecycleManager._State.ENTERED_FOR_TRAINING
            return self

        def __exit__(self, type_, value, traceback):
            assert self.lifecycle_manager._state is \
                TrainerLifecycleManager._State.ENTERED_FOR_TRAINING
            return self.lifecycle_manager.release(
                type_, value, traceback)

        def train(self, dataset: VisionPipelineDataset,
                  val_dataset: Optional[VisionPipelineDataset] = None):
            assert self.lifecycle_manager._state is \
                TrainerLifecycleManager._State.ENTERED_FOR_TRAINING
            self.lifecycle_manager.implementation.train(
                dataset, val_dataset=val_dataset)

    def prepare_for_training(self):
        assert self._state is \
            TrainerLifecycleManager._State.INITIALIZED
        self.implementation.prepare_for_training(self._metadata)
        prepared = TrainerLifecycleManager._PreparedForTraining(self)
        self._state = \
            TrainerLifecycleManager._State.PREPARED_FOR_TRAINING
        return prepared

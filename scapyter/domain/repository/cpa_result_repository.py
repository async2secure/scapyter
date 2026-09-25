from abc import ABC, abstractmethod

from scapyter.domain.value_object import CpaByteResult


class CpaResultRepository(ABC):
    @abstractmethod
    def save(self, result: CpaByteResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, byte_index: int) -> CpaByteResult:
        raise NotImplementedError

    @abstractmethod
    def exists(self, byte_index: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_all(self) -> list[CpaByteResult]:
        raise NotImplementedError

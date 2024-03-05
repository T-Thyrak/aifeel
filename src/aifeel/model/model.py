from __future__ import annotations

from os import PathLike
from typing import TypeVar, Generic
from typing_extensions import Self
from abc import ABC, abstractmethod
from dill import dump


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Model(ABC, Generic[T, U]):
    def __init__(self):
        super().__init__()

        # self.is_fitted_ = False

    def fit(self: Self, X: T, y: U) -> Self:
        fitted_model = self._fit(X, y)
        self.is_fitted_ = True
        return fitted_model

    def predict(self: Self, X: T) -> U:
        # if not self.is_fitted_:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet.")
        return self._predict(X)

    def save(self: Self, path: str | PathLike) -> None:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet.")
        self._save(path)

    def _save(self: Self, path: str | PathLike) -> None:
        with open(path, "wb") as f:
            dump(self, f)

    @abstractmethod
    def _fit(self: Self, X: T, y: U) -> Self:
        pass

    @abstractmethod
    def _predict(self: Self, X: T) -> U:
        pass

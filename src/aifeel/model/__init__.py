from __future__ import annotations

from os import PathLike
from typing import Self, TypeVar, Generic
from abc import ABC, abstractmethod
from dill import dump

try:
    import pandas as pd
except ImportError:
    _has_pandas = False
else:
    _has_pandas = True

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Model(ABC, Generic[T, U]):
    def __init__(self):
        super().__init__()

        self._is_fitted = False

    def fit(self: Self, X: T, y: U) -> Self:
        fitted_model = self._fit(X, y)
        self._is_fitted = True
        return fitted_model

    def predict(self: Self, X: T) -> U:
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet.")
        return self._predict(X)

    def save(self: Self, path: str | PathLike) -> None:
        if not self._is_fitted:
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


if _has_pandas:

    class ExampleModel(Model[pd.Series, pd.Series]):
        def _fit(self, X: pd.Series, y: pd.Series) -> ExampleModel:
            return self

        def _predict(self, X: pd.Series) -> pd.Series:
            return X

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    _has_numpy = True
else:
    try:
        import numpy as np

        _has_numpy = True
    except ImportError:
        _has_numpy = False


if TYPE_CHECKING:
    import pandas as pd

    _has_pandas = True
else:
    try:
        import pandas as pd

        _has_pandas = True
    except ImportError:
        _has_pandas = False

if TYPE_CHECKING:
    import tensorflow as tf
    import keras
    from keras import layers
    from keras.models import Sequential

    _has_tensorflow = True
else:
    try:
        import tensorflow as tf
        import keras
        from keras import layers
        from keras.models import Sequential

        _has_tensorflow = True
    except ImportError:
        _has_tensorflow = False

if TYPE_CHECKING:
    import sklearn
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.metrics import accuracy_score

    _has_sklearn = True
else:
    try:
        import sklearn
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.metrics import accuracy_score

        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

from aifeel.model.model import Model

if not _has_numpy:
    raise ImportError("numpy is not installed")
if not _has_pandas:
    raise ImportError("pandas is not installed")
if not _has_tensorflow:
    raise ImportError("tensorflow is not installed")
if not _has_sklearn:
    raise ImportError("sklearn is not installed")

Float = float | np.float16 | np.float32 | np.float64


class NNClassifier(Model[np.ndarray, np.ndarray], ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        threshold: float = 0.5,
        epochs=10,
        batch_size=32,
        lr=0.001,
        early_stopping=False,
        patience=3,
    ):
        """Neural Network Classifier.

        Note: Expects a non-compiled model, if you decide to give a model."""
        super().__init__()
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping = early_stopping
        self.patience = patience
        # self.input_dim = input_dim

        # if model is None:
        #     self.model = Sequential(
        #         [
        #             layers.Dense(512, input_dim=self.input_dim, activation="relu"),
        #             layers.Dropout(0.5),
        #             layers.Dense(256, activation="relu"),
        #             layers.Dropout(0.5),
        #             # layers.Dense(256),
        #             # layers.Dropout(0.5),
        #             # layers.Dense(128, activation="relu"),
        #             # layers.Dropout(0.5),
        #             # layers.Dense(256, activation="relu"),
        #             # layers.Dropout(0.5),
        #             # layers.Dense(512, activation="relu"),
        #             # layers.Dropout(0.5),
        #             # layers.Dense(1, activation="sigmoid"),
        #             # layers.Embedding(input_dim, 64, input_length=input_dim),
        #             # layers.LSTM(64, return_sequences=True, activation="relu"),
        #             # layers.Dropout(0.5),
        #             # layers.LSTM(64, activation="relu"),
        #             # layers.Dropout(0.5),
        #             layers.Dense(1, activation="sigmoid"),
        #         ]
        #     )
        # else:
        #     self.model = model

        # optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # self.model.compile(
        #     loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        # )

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NNClassifier:
        self.model_ = Sequential(
            [
                layers.Dense(512, input_dim=X.shape[1], activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.optimizer_ = keras.optimizers.Adam(learning_rate=self.lr)

        self.model_.compile(
            loss="binary_crossentropy", optimizer=self.optimizer_, metrics=["accuracy"]
        )

        if self.early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            )
            self.model_.fit(
                X,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stopping],
            )
        else:
            self.model_.fit(
                X,
                y,
                validation_split=0.1,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # return np.array(self.model.predict(X).flatten())
        return (self.model_.predict(X) > self.threshold).astype("int32").flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pos_pred = self.model_.predict(X).flatten()
        neg_pred = 1 - pos_pred

        result = [[neg_pred[i], pos_pred[i]] for i in range(len(pos_pred))]
        # print(result)

        return np.array(result)

    # def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> Float:
    #     # for cross validation purposes
    #     return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {
            "threshold": self.threshold,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
        }

    def set_params(self, **params):
        self.threshold = params["threshold"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.lr = params["lr"]
        self.early_stopping = params["early_stopping"]
        self.patience = params["patience"]
        return self

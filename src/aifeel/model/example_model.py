from typing import TYPE_CHECKING
from typing_extensions import Self

from aifeel.model.model import Model


# watch and learn
# this is how you do an optional import
if TYPE_CHECKING:
    import pandas as pd

    _has_pandas = True
else:
    try:
        import pandas as pd
    except ImportError:

        _has_pandas = False
    else:
        _has_pandas = True


if _has_pandas:

    class ExampleModel(Model[pd.Series, pd.Series]):
        def _fit(self, X: pd.Series, y: pd.Series) -> Self:
            return self

        def _predict(self, X: pd.Series) -> pd.Series:
            return X

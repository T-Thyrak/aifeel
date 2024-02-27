from pathlib import Path
from typing import TYPE_CHECKING

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
    import numpy as np

    _has_numpy = True
else:
    try:
        import numpy as np

        _has_numpy = True
    except ImportError:
        _has_numpy = False


from .preprocess import preprocess_corpus, preprocess_text

CORPORA_DIR = Path(__file__).parent.parent / "corpora"


def read_corpus(key: str) -> list[str]:
    """Read corpus from file."""

    # does file exist?
    file = CORPORA_DIR / f"{key}.txt"
    if not file.exists():
        # perhaps missing .txt?
        file = CORPORA_DIR / f"{key}"
        if not file.exists():
            # maybe it's supposed to be a path relative to where the script is running?
            file = Path(key)
            if not file.exists():
                raise FileNotFoundError(f"Could not find corpus file {key}")

    # read file
    with open(file, "r", encoding="latin-1") as f:
        return [sentence.strip() for sentence in f.readlines()]


def gen_dataframe(
    positive_corpus: list[str],
    negative_corpus: list[str],
    shuffle: bool = True,
    random_state: int | None = None,
) -> "pd.DataFrame":
    """Generate a Pandas DataFrame object from the positive and negative corpora."""
    if not _has_pandas:
        raise ImportError(
            "Pandas is not installed. Try reinstalling with `pip install -e '.[pandas]'`."
        )
    if not _has_numpy:
        raise ImportError(
            "Numpy is not installed. Try reinstalling with `pip install -e '.[pandas]'`."
        )

    data = np.array(
        [[1, review] for review in positive_corpus]
        + [[0, review] for review in negative_corpus]
    )
    if shuffle:
        if random_state is None:
            np.random.shuffle(data)
        else:
            randomizer = np.random.RandomState(random_state)
            randomizer.shuffle(data)

    return pd.DataFrame(data, columns=["tag", "review"])

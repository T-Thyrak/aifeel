from pathlib import Path


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

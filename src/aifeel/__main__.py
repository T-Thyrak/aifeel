import logging

from rich import print
from rich.logging import RichHandler

from aifeel.util import gen_dataframe, read_corpus
from aifeel.util.preprocess import preprocess_text

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[
        RichHandler(markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
    ],
)


negative_corpus, positive_corpus = read_corpus("negative-reviews"), read_corpus(
    "positive-reviews"
)
negative_words, positive_words = read_corpus("negative-words"), read_corpus(
    "positive-words"
)

df = gen_dataframe(positive_corpus, negative_corpus, random_state=42)
df["clean_review"] = df["review"].apply(preprocess_text)
print(df.head())

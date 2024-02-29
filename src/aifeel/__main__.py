import logging

import numpy as np
from rich import print
from rich.logging import RichHandler

from aifeel.util import gen_dataframe, read_corpus
from aifeel.util.feature_extraction import extract_features, feature_to_vector
from aifeel.util.preprocess import preprocess_text

from sklearn.feature_extraction.text import CountVectorizer

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
negative_words, positive_words = set(read_corpus("negative-words")), set(
    read_corpus("positive-words")
)

df = gen_dataframe(positive_corpus, negative_corpus, random_state=42)
df["clean_review"] = df["review"].apply(preprocess_text)

cv = CountVectorizer(max_features=1000 - 8)  # 6 primary features + 2 extra features
cv.fit(df["clean_review"])


def vectorizer(review):
    result = cv.transform([review])
    return result.toarray()[0].tolist()  # type: ignore
    #        ^      ^          ^
    #   spmatrix ndarray    list
    # it definitely exists, type hinter is just bad


df["features"] = df["clean_review"].apply(
    extract_features, args=(positive_words, negative_words), vectorizer=vectorizer
)
df["feature_vector"] = df["features"].apply(feature_to_vector, vectorizer=True)
fv = np.array(df.head(5).iloc[0]["feature_vector"])

print(fv.shape)

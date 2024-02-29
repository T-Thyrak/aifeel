from math import log
import re
from typing import Callable, Iterable, Optional
from nltk import ngrams
from aifeel.util import read_corpus

Number = int | float
Features = dict[str, Number | list[int]]


def extract_features(
    review: str,
    positive_words: set[str],
    negative_words: set[str],
    *,
    vectorizer: Optional[Callable[[str], list[int]]] = None,
    count_pseudo_negative_feature: bool = True,
    count_pseudo_positive_feature: bool = True,
) -> Features:
    feature: Features = {}
    words = set(re.findall(r"\b\w+\b", review.lower()))

    feature["negative_count"] = len(words & negative_words)
    feature["positive_count"] = len(words & positive_words)
    feature["contains_no"] = int("no" in words)
    feature["pronoun_count"] = len(words & {"i", "me", "my", "you", "your"})
    feature["contains_exclamation"] = int("!" in review)
    feature["log_length"] = log(len(review) + 1)

    if count_pseudo_negative_feature:
        feature["count_pseudo_negative"] = sum(
            1
            for prev_word, next_word in ngrams(review.split(), 2)
            if prev_word in ["not", "no"] and next_word in positive_words
        )

    if count_pseudo_positive_feature:
        feature["count_pseudo_positive"] = sum(
            1
            for prev_word, next_word in ngrams(review.split(), 2)
            if prev_word in ["not", "no"] and next_word in negative_words
        )

    if vectorizer:
        vector = vectorizer(review)
        feature["vector"] = vector

    return feature


def extract_features_n(
    reviews: Iterable[str],
    negative_words_corpus: str | list[str] = "negative-words",
    positive_words_corpus: str | list[str] = "positive-words",
    *,
    vectorizer: Optional[Callable[[str], list[int]]] = None,
    count_pseudo_negative_feature: bool = True,
    count_pseudo_positive_feature: bool = True,
) -> list[Features]:
    features: list[Features] = []

    if type(negative_words_corpus) == str:
        negative_words = set(read_corpus(negative_words_corpus))
    else:
        negative_words = set(negative_words_corpus)

    if type(positive_words_corpus) == str:
        positive_words = set(read_corpus(positive_words_corpus))
    else:
        positive_words = set(positive_words_corpus)

    for review in reviews:
        feature = extract_features(
            review,
            positive_words,
            negative_words,
            vectorizer=vectorizer,
            count_pseudo_negative_feature=count_pseudo_negative_feature,
            count_pseudo_positive_feature=count_pseudo_positive_feature,
        )

        features.append(feature)

    return features


def feature_to_vector(
    features: Features,
    *,
    vectorizer: bool = False,
    count_pseudo_negative_feature: bool = True,
    count_pseudo_positive_feature: bool = True,
) -> list[Number]:
    return (
        [
            features["negative_count"],
            features["positive_count"],
            features["contains_no"],
            features["pronoun_count"],
            features["contains_exclamation"],
            features["log_length"],
        ]
        + ([features["count_pseudo_negative"]] if count_pseudo_negative_feature else [])
        + ([features["count_pseudo_positive"]] if count_pseudo_positive_feature else [])
        + (features["vector"] if vectorizer else [])
    )  # type: ignore
    # i know what i'm doing

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

HTML_TAGS_PATTERN = re.compile(r"<.*?>")
URI_PATTERN = re.compile(r"(.*?)\:\/\/(www\.)?(.*?)\/(.*?)")
NORMAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s!]")
stopwords_en = stopwords.words("english")

# specifically keep i, me, my, you, your
if "i" in stopwords_en:
    stopwords_en.remove("i")
if "me" in stopwords_en:
    stopwords_en.remove("me")
if "my" in stopwords_en:
    stopwords_en.remove("my")
if "you" in stopwords_en:
    stopwords_en.remove("you")

def lemmatize_text(text: str) -> str:
    """Lemmatize text."""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

import re

def handle_negations(text: str) -> str:
    """Handle negations in text."""
    transformed = re.sub(r'\b(?:not|isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|shan\'t|shouldn\'t|can\'t|cannot|couldn\'t|mustn\'t)\b[\w\s]+?(?=\b(?:not|isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|shan\'t|shouldn\'t|can\'t|cannot|couldn\'t|mustn\'t)\b|[^\w\s])', 
                         lambda match: re.sub(r'(\s+)(\w+)', r'\1NOT_\2', match.group(0)), 
                         text,
                         flags=re.IGNORECASE)
    return transformed


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    text = re.sub(HTML_TAGS_PATTERN, "", text)
    return text


def remove_links(text: str) -> str:
    """Remove links from text."""
    text = re.sub(URI_PATTERN, "", text)
    return text


def remove_special_characters(text: str) -> str:
    """Remove special characters from text."""
    text = re.sub(NORMAL_CHARS_PATTERN, " ", text)
    return text


def remove_extra_spaces(text: str) -> str:
    """Remove extra spaces from text."""
    text = re.sub(r"\s+", " ", text)
    return text


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text."""
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords_en]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Preprocess text."""
    text = remove_html_tags(text)
    text = remove_links(text)
    text = remove_special_characters(text)
    text = remove_extra_spaces(text)
    text = to_lowercase(text)
    text = handle_negations(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def preprocess_corpus(corpus: list[str]) -> list[str]:
    """Preprocess corpus."""
    return [preprocess_text(sentence) for sentence in corpus]

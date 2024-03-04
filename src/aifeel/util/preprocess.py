import re
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

HTML_TAGS_PATTERN = re.compile(r"<.*?>")
URI_PATTERN = re.compile(r"(.*?)\:\/\/(www\.)?(.*?)\/(.*?)")
NORMAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s!]")
stopwords_en = stopwords.words("english")

# specifically keep i, me, my, you, your
keep_words = [
    "i",
    "me",
    "my",
    "you",
    "your",
    "no",
    "not",
    "nor",
    "have",
    "has",
    "had",
    "should",
    "shouldn't",
    "haven't",
    "hasn't",
    "hadn't",
    "can",
    "can't",
    "must",
    "mustn't",
    "would",
    "wouldn't",
    "will",
    "won't",
    "shall",
    "shan't",
    "might",
]

for word in keep_words:
    if word in stopwords_en:
        stopwords_en.remove(word)

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return "a"
    elif treebank_tag.startswith("V"):
        return "v"
    elif treebank_tag.startswith("N"):
        return "n"
    elif treebank_tag.startswith("R"):
        return "r"
    else:
        return "n"


def lemmatize_text(text: str) -> str:
    """Lemmatize text."""
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
    ]
    return " ".join(tokens)


# def handle_negations(text: str) -> str:
#     """Handle negations in text."""
#     # transformed = re.sub(
#     #     r"\b(?:not|isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|shan\'t|shouldn\'t|can\'t|cannot|couldn\'t|mustn\'t)\b[\w\s]+?(?=\b(?:not|isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|shan\'t|shouldn\'t|can\'t|cannot|couldn\'t|mustn\'t)\b|[^\w\s])",
#     #     lambda match: re.sub(r"(\s+)(\w+)", r"\1NOT_\2", match.group(0)),
#     #     text,
#     #     flags=re.IGNORECASE,
#     # )

#     return transformed


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


# def break_contractions(text: str) -> str:
#     """Break contractions in text."""
#     mappings = {
#         "i'm": "i am",
#         "i'd": "i would",
#         "i'll": "i will",
#         "i've": "i have",
#         "you're": "you are",
#         "you'd": "you would",
#         "you'll": "you will",
#         "you've": "you have",
#         "he's": "he is",
#         "he'd": "he would",
#         "he'll": "he will",

#     }


def handle_negations(text: str) -> str:
    """Negation handling."""
    tokens = nltk.word_tokenize(text)
    new_tokens = [tokens[0]]
    for pt, nt in ngrams(tokens, 2):
        if pt.endswith("n't") or pt in ["no", "not", "nor"]:
            new_tokens.append("NOT_" + nt)
        else:
            new_tokens.append(nt)

    return " ".join(new_tokens)


def preprocess_text(text: str) -> str:
    """Preprocess text."""
    og_text = text
    text = remove_html_tags(text)
    text = remove_links(text)
    text = remove_special_characters(text)
    text = remove_extra_spaces(text)
    text = to_lowercase(text).strip()
    if not text:
        return text
    # text = handle_negations(text)
    text = handle_negations(text)
    text = lemmatize_text(text)
    # text = remove_stopwords(text)
    return text


def preprocess_corpus(corpus: list[str]) -> list[str]:
    """Preprocess corpus."""
    return [preprocess_text(sentence) for sentence in corpus]

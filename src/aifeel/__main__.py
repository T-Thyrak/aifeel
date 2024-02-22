from aifeel.util import read_corpus


def __init():
    import logging
    from rich import print
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[
            RichHandler(markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
        ],
    )


__init()


negative_corpus, positive_corpus = read_corpus("negative-reviews"), read_corpus(
    "positive-reviews"
)
negative_words, positive_words = read_corpus("negative-words"), read_corpus(
    "positive-words"
)

print(negative_words)

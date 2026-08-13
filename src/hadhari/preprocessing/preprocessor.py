import re

import emoji

_CHAR_MAP = str.maketrans({
    "أ": "ا",  # noqa: RUF001
    "إ": "ا",  # noqa: RUF001
    "آ": "ا",  # noqa: RUF001
    "ى": "ي",
    "ة": "ه",  # noqa: RUF001
    "ؤ": "و",
    "ئ": "ي",
    "ـ": "",
})


# Patterns
_DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0652]")
_REPEATED_CHAR_PATTERN = re.compile(r"(.)\1+")
_URL_PATTERN = re.compile(
    r"""(?xi)
    \b
    (?:https?:\/\/)?               # optional http or https
    (?:www\.)?                     # optional www.
    [a-z0-9\-._~%]+                # domain or subdomain
    \.
    [a-z]{2,}                      # TLD
    (?:[\/?#][^\s]*)?              # optional path/query
    \b
    """,
)

_PHONE_PATTERN = re.compile(r"\b\d{8,15}\b")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s<>]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_arabic_letters(text: str) -> str:
    return text.translate(_CHAR_MAP)


def remove_diacritics(text: str) -> str:
    return _DIACRITICS_PATTERN.sub("", text)


def reduce_repeated_characters(text: str) -> str:
    return _REPEATED_CHAR_PATTERN.sub(r"\1", text)


def replace_urls(text: str) -> str:
    return _URL_PATTERN.sub(" <url> ", text)


def replace_phone_numbers(text: str) -> str:
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return _PHONE_PATTERN.sub(" <phone> ", text)


def replace_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace=" <emoji> ")


def remove_punctuation(text: str) -> str:
    return _PUNCTUATION_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = normalize_arabic_letters(text)
    text = remove_diacritics(text)
    text = replace_urls(text)
    text = replace_phone_numbers(text)
    text = replace_emojis(text)
    text = reduce_repeated_characters(text)
    text = remove_punctuation(text)

    return normalize_whitespace(text)


def preprocess_texts(texts: list[str]) -> list[str]:
    """Apply clean_text to each text in the input.

    Used as the preprocessing step in the sklearn Pipeline via FunctionTransformer.
    Must be importable at deserialization time.

    Returns
    -------
    list[str]
        The cleaned texts.

    """
    return [clean_text(text) for text in texts]

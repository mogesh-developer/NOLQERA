import re

from ..utils.text_utils import validate_text

def remove_url(text : str) -> str:
    """Remove the URLs from text"""
    return re.sub(r"https?://\S+|www\.\S+","",text)

def remove_html_tag(text: str) -> str:
    """Remove HTML tags from text"""
    clean = re.compile(r'<.*?>')
    return re.sub(clean, '',text)

def remove_mentions(text: str) -> str:
    """Remove mentions from text"""
    return re.sub(r"@\w+","",text)

def remove_numbers(text: str) -> str:
    """Remove numbers from text"""
    return re.sub(r"\d+","",text)

def remove_special_char(text: str) -> str:
    """Remove special characters from text"""
    return re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", "", text)

remove_urls = remove_url


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace from text"""
    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text: str) -> str:
    """Preprocess text"""
    text = remove_url(text)
    text = remove_html_tag(text)
    text = remove_mentions(text)
    text = remove_numbers(text)
    text = remove_special_char(text)
    text = remove_extra_whitespace(text)
    return text

def remove_email_addresses(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r"\S+@\S+\.\S+", "", text)

def remove_urls(text: str) -> str:
    """Remove URLs from text."""

    validate_text(text)

    return re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text,
    )


def remove_email_addresses(text: str) -> str:
    """Remove email addresses from text."""

    validate_text(text)

    return re.sub(
        r"\S+@\S+\.\S+",
        "",
        text,
    )


def remove_extra_whitespace(text: str) -> str:
    """Replace multiple whitespace characters with a single space."""

    validate_text(text)

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()
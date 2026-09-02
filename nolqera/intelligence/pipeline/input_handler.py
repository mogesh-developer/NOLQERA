from nolqera.preprocessing.cleaner import (
    remove_email_addresses,
    remove_extra_whitespace,
    remove_urls,
)


class InputHandler:
    """
    Validates and safely preprocesses raw input before it
    enters the NOLQERA processing pipeline.

    This stage intentionally preserves:
    - capitalization
    - punctuation
    - sentence boundaries
    - original wording

    It only removes low-value structural noise and normalizes
    whitespace.
    """

    def handle(self, raw_input: str) -> str:
        if not isinstance(raw_input, str):
            raise TypeError("raw_input must be a string")

        if not raw_input.strip():
            raise ValueError("raw_input cannot be empty")

        # -----------------------------------------------------
        # 1. Remove URLs
        # -----------------------------------------------------

        text = remove_urls(raw_input)

        # -----------------------------------------------------
        # 2. Remove email addresses
        # -----------------------------------------------------

        text = remove_email_addresses(text)

        # -----------------------------------------------------
        # 3. Normalize whitespace
        # -----------------------------------------------------

        text = remove_extra_whitespace(text)

        # -----------------------------------------------------
        # 4. Validate that preprocessing did not eliminate
        #    the complete input.
        # -----------------------------------------------------

        if not text:
            raise ValueError(
                "raw_input contains no usable text after preprocessing"
            )

        return text
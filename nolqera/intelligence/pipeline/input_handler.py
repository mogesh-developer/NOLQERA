class InputHandler:
    """
    Validates and normalizes raw input before it enters
    the NOLQERA processing pipeline.
    """

    def handle(self, raw_input: str) -> str:
        if not isinstance(raw_input, str):
            raise TypeError("raw_input must be a string")

        if not raw_input.strip():
            raise ValueError("raw_input cannot be empty")

        # Normalize surrounding and repeated whitespace.
        return " ".join(raw_input.split())
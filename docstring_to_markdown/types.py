from typing_extensions import Protocol


class Converter(Protocol):

    def convert(self, docstring: str) -> str:
        """Convert given docstring to markdown."""

    def can_convert(self, docstring: str) -> bool:
        """Check if conversion to markdown can be performed."""

    # The higher the priority, the sooner the conversion
    # with this converter will be attempted.
    priority: int

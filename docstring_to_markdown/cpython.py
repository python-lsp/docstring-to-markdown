from typing import Union, List
from re import fullmatch

from .types import Converter
from ._utils import escape_markdown


def _is_cpython_signature_line(line: str) -> bool:
    """CPython uses signature lines in the following format:

    str(bytes_or_buffer[, encoding[, errors]]) -> str
    """
    return fullmatch(r'\w+\(\S*(, \S+)*(\[, \S+\])*\)\s--?>\s.+', line) is not None


def cpython_to_markdown(text: str) -> Union[str, None]:
    signature_lines: List[str] = []
    other_lines: List[str] = []
    for line in text.splitlines():
        if not other_lines and _is_cpython_signature_line(line):
            signature_lines.append(line)
        elif not signature_lines:
            return None
        elif line.startswith('    '):
            signature_lines.append(line)
        else:
            other_lines.append(line)
    return '\n'.join([
        '```',
        '\n'.join(signature_lines),
        '```',
        escape_markdown('\n'.join(other_lines))
    ])


def looks_like_cpython(text: str) -> bool:
    return cpython_to_markdown(text) is not None


class CPythonConverter(Converter):

    priority = 10

    def __init__(self) -> None:
        self._last_docstring: Union[str, None] = None
        self._converted: Union[str, None] = None

    def can_convert(self, docstring):
        self._last_docstring = docstring
        self._converted = cpython_to_markdown(docstring)
        return self._converted is not None

    def convert(self, docstring):
        if docstring != self._last_docstring:
            self._last_docstring = docstring
            self._converted = cpython_to_markdown(docstring)
        return self._converted


__all__ = ['looks_like_cpython', 'cpython_to_markdown', 'CPythonConverter']

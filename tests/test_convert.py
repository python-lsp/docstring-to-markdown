from contextlib import contextmanager
from docstring_to_markdown import convert, UnknownFormatError
from docstring_to_markdown.types import Converter
from docstring_to_markdown.cpython import CPythonConverter
from importlib_metadata import EntryPoint, entry_points, distribution
from unittest.mock import patch
import docstring_to_markdown
import pytest

CPYTHON = """\
bool(x) -> bool

Returns True when the argument x is true, False otherwise.\
"""


CPYTHON_MD = """\
```
bool(x) -> bool
```

Returns True when the argument x is true, False otherwise.\
"""

GOOGLE = """Do **something**.

Args:
    a: some arg
    b: some arg
"""

GOOGLE_MD = """Do **something**.

#### Args

- `a`: some arg
- `b`: some arg
"""


RST = "Please see `this link<https://example.com>`__."
RST_MD = "Please see [this link](https://example.com)."


def test_convert_cpython():
    assert convert(CPYTHON) == CPYTHON_MD


def test_convert_plain_text():
    assert convert('This is a sentence.') == 'This is a sentence.'


def test_convert_google():
    assert convert(GOOGLE) == GOOGLE_MD


def test_convert_rst():
    assert convert(RST) == RST_MD


def test_unknown_format():
    with pytest.raises(UnknownFormatError):
        convert('ARGS [arg1, arg2] RETURNS: str OR None')


class HighPriorityConverter(Converter):
    priority = 120

    def convert(self, docstring):
        return "HighPriority"

    def can_convert(self, docstring):
        return True


class MockEntryPoint(EntryPoint):
    def load(self):
        return globals()[self.attr]

    dist = None


class DistMockEntryPoint(MockEntryPoint):
    # Pretend it is contributed by `pytest`.
    # It could be anything else, but `pytest`
    # is guaranteed to be installed during tests.
    dist = distribution('pytest')


class CustomCPythonConverter(CPythonConverter):
    priority = 10

    def convert(self, docstring):
        return 'CustomCPython'

    def can_convert(self, docstring):
        return True


@contextmanager
def custom_entry_points(entry_points):
    old = docstring_to_markdown._CONVERTERS
    docstring_to_markdown._CONVERTERS = None
    with patch.object(docstring_to_markdown, 'entry_points', return_value=entry_points):
        yield
    docstring_to_markdown._CONVERTERS = old


def test_adding_entry_point():
    original_entry_points = entry_points(group="docstring_to_markdown")
    mock_entry_point = MockEntryPoint(
        name='high-priority-converter',
        group='docstring_to_markdown',
        value="here:HighPriorityConverter",
    )
    with custom_entry_points([*original_entry_points, mock_entry_point]):
        assert convert('test') == 'HighPriority'


def test_replacing_entry_point():
    assert convert(CPYTHON) == CPYTHON_MD
    original_entry_points = entry_points(group="docstring_to_markdown")
    mock_entry_point = DistMockEntryPoint(
        name='cpython',
        group='docstring_to_markdown',
        value="here:CustomCPythonConverter",
    )
    with custom_entry_points([*original_entry_points, mock_entry_point]):
        assert convert('test') == 'test'
        assert convert(GOOGLE) == GOOGLE_MD
        assert convert(RST) == RST_MD
        assert convert(CPYTHON) == 'CustomCPython'

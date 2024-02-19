from docstring_to_markdown import convert, UnknownFormatError
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

import pytest
from docstring_to_markdown.cpython import looks_like_cpython, cpython_to_markdown, CPythonConverter

BOOL = """\
bool(x) -> bool

Returns True when the argument x is true, False otherwise.\
"""

BOOL_MD = """\
```
bool(x) -> bool
```

Returns True when the argument x is true, False otherwise.\
"""

BYTES = """\
bytes(iterable_of_ints) -> bytes
bytes(string, encoding[, errors]) -> bytes
bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer
bytes(int) -> bytes object of size given by the parameter initialized with null bytes
bytes() -> empty bytes object

Construct an immutable array of bytes from:
  - an iterable yielding integers in range(256)
  - a text string encoded using the specified encoding
  - any object implementing the buffer API.
  - an integer\
"""

COLLECTIONS_DEQUEUE = """\
deque([iterable[, maxlen]]) --> deque object

A list-like sequence optimized for data accesses near its endpoints.\
"""

DICT = """\
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)\
"""

STR = """\
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).\
"""

STR_MD = """\
```
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str
```

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.\\_\\_str\\_\\_() (if defined)
or repr(object).\
"""


@pytest.mark.parametrize("text", [BYTES, STR, DICT, BOOL, COLLECTIONS_DEQUEUE])
def test_accepts_cpython_docstrings(text):
    assert looks_like_cpython(text) is True


@pytest.mark.parametrize("text", [
    "[link label](https://link)",
    "![image label](https://source)",
    "Some **bold** text",
    "More __bold__ text",
    "Some *italic* text",
    "More _italic_ text",
    "This is a sentence.",
    "Exclamation!",
    "Can I ask a question?",
    "Let's send an e-mail",
    "Parentheses (are) fine (really)",
    "Double \"quotes\" and single 'quotes'"
])
def test_rejects_markdown_and_plain_text(text):
    assert looks_like_cpython(text) is False


def test_conversion_bool():
    assert cpython_to_markdown(BOOL) == BOOL_MD


def test_conversion_str():
    assert cpython_to_markdown(STR) == STR_MD


def test_convert():
    converter = CPythonConverter()
    assert converter.can_convert(BOOL)
    assert not converter.can_convert('this is plain text')
    assert converter.convert(BOOL) == BOOL_MD

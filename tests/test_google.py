import pytest

from docstring_to_markdown.google import google_to_markdown, looks_like_google, GoogleConverter

BASIC_EXAMPLE = """Do **something**.

Some more detailed description.

Args:
    a: some arg
    b: some arg

Returns:
    Same *stuff*
"""

BASIC_EXAMPLE_MD = """Do **something**.

Some more detailed description.

#### Args

- `a`: some arg
- `b`: some arg

#### Returns

- Same *stuff*
"""

ESCAPE_MAGIC_METHOD = """Example.

Args:
    a: see __init__.py
"""

ESCAPE_MAGIC_METHOD_MD = """Example.

#### Args

- `a`: see \\_\\_init\\_\\_.py
"""

PLAIN_SECTION = """Example.

Args:
    a: some arg

Note:
    Do not use this.

Example:
    Do it like this.
"""

PLAIN_SECTION_MD = """Example.

#### Args

- `a`: some arg

#### Note

Do not use this.

#### Example

Do it like this.
"""

MULTILINE_ARG_DESCRIPTION = """Example.

Args:
    a (str): This is a long description
             spanning over several lines
        also with broken indentation
    b (str): Second arg
    c (str):
        On the next line
        And also multiple lines
"""

MULTILINE_ARG_DESCRIPTION_MD = """Example.

#### Args

- `a (str)`: This is a long description
             spanning over several lines
             also with broken indentation
- `b (str)`: Second arg
- `c (str)`: On the next line
             And also multiple lines
"""

GOOGLE_CASES = {
    "basic example": {
        "google": BASIC_EXAMPLE,
        "md": BASIC_EXAMPLE_MD,
    },
    "escape magic method": {
        "google": ESCAPE_MAGIC_METHOD,
        "md": ESCAPE_MAGIC_METHOD_MD,
    },
    "plain section": {
        "google": PLAIN_SECTION,
        "md": PLAIN_SECTION_MD,
    },
    "multiline arg description": {
        "google": MULTILINE_ARG_DESCRIPTION,
        "md": MULTILINE_ARG_DESCRIPTION_MD,
    },
}


@pytest.mark.parametrize(
    "google",
    [case["google"] for case in GOOGLE_CASES.values()],
    ids=GOOGLE_CASES.keys(),
)
def test_looks_like_google_recognises_google(google):
    assert looks_like_google(google)


def test_looks_like_google_ignores_plain_text():
    assert not looks_like_google("This is plain text")
    assert not looks_like_google("See Also\n--------\n")


@pytest.mark.parametrize(
    "google,markdown",
    [[case["google"], case["md"]] for case in GOOGLE_CASES.values()],
    ids=GOOGLE_CASES.keys(),
)
def test_google_to_markdown(google, markdown):
    assert google_to_markdown(google) == markdown


def test_converter():
    converter = GoogleConverter()
    assert converter.can_convert(BASIC_EXAMPLE)
    assert not converter.can_convert("This is plain text")
    assert converter.convert(BASIC_EXAMPLE) == BASIC_EXAMPLE_MD

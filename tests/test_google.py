from docstring_to_markdown.google import google_to_markdown, looks_like_google

BASIC_EXAMPLE = """Do **something**.

Args:
    a: some arg
    b: some arg

Returns:
    Same *stuff*
"""

BASIC_EXAMPLE_MD = """Do **something**.

# Args

- a: some arg
- b: some arg

# Returns

- Same *stuff*
"""


def test_looks_like_google_recognises_google():
    assert looks_like_google(BASIC_EXAMPLE)


def test_looks_like_google_ignores_plain_text():
    assert not looks_like_google("This is plain text")
    assert not looks_like_google("See Also\n--------\n")


def test_google_to_markdown():
    assert google_to_markdown(BASIC_EXAMPLE) == BASIC_EXAMPLE_MD

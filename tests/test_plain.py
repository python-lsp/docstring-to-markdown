import pytest
from docstring_to_markdown.plain import looks_like_plain_text, plain_text_to_markdown, PlainTextConverter


@pytest.mark.parametrize("text", [
    "This is a sentence.",
    "Exclamation!",
    "Can I ask a question?",
    "Let's send an e-mail",
    "Parentheses (are) fine (really)",
    "Double \"quotes\" and single 'quotes'"
])
def test_accepts_english(text):
    assert looks_like_plain_text(text) is True


@pytest.mark.parametrize("text", [
    "[link label](https://link)",
    "![image label](https://source)",
    "Some **bold** text",
    "More __bold__ text",
    "Some *italic* text",
    "More _italic_ text"
])
def test_rejects_markdown(text):
    assert looks_like_plain_text(text) is False


@pytest.mark.parametrize("text", [
    "def test():",
    "print(123)",
    "func(arg)",
    "2 + 2",
    "var['test']",
    "x = 'test'"
])
def test_rejects_code(text):
    assert looks_like_plain_text(text) is False


def test_conversion():
    assert plain_text_to_markdown("test") == "test"


def test_convert():
    converter = PlainTextConverter()
    assert not converter.can_convert('def test():')
    assert converter.can_convert('this is plain text')
    assert converter.convert('test') == 'test'

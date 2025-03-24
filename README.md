# docstring-to-markdown

[![tests](https://github.com/python-lsp/docstring-to-markdown/workflows/tests/badge.svg)](https://github.com/python-lsp/docstring-to-markdown/actions?query=workflow%3A%22tests%22)
![CodeQL](https://github.com/python-lsp/docstring-to-markdown/workflows/CodeQL/badge.svg)
[![pypi-version](https://img.shields.io/pypi/v/docstring-to-markdown.svg)](https://python.org/pypi/docstring-to-markdown)

On the fly conversion of Python docstrings to markdown

- Python 3.7+ (tested on 3.8 up to 3.13)
- can recognise reStructuredText and convert multiple of its features to Markdown
- since v0.13 includes initial support for Google-formatted docstrings

### Installation

```bash
pip install docstring-to-markdown
```

### Example

Convert reStructuredText:

```python
>>> import docstring_to_markdown
>>> docstring_to_markdown.convert(':math:`\\sum`')
'$\\sum$'
```

When given the format cannot be recognised an exception will be raised:

```python
>>> docstring_to_markdown.convert('\\sum')
Traceback (most recent call last):
    raise UnknownFormatError()
docstring_to_markdown.UnknownFormatError
```

### Extensibility

`docstring_to_markdown` entry point group allows to add custom converters which follow the `Converter` protocol.
The built-in converters can be customized by providing entry point with matching name.

### Development

```bash
pip install -e .
pytest
```

from .rst import looks_like_rst, rst_to_markdown


class UnknownFormatError(Exception):
    pass


def convert(docstring: str) -> str:
    if looks_like_rst(docstring):
        return rst_to_markdown(docstring)
    raise UnknownFormatError()

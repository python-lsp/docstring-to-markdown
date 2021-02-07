from docstring_to_markdown.rst import looks_like_rst, rst_to_markdown


SEE_ALSO = """
See Also
--------
DataFrame.from_records : Constructor from tuples, also record arrays.
read_table : Read general delimited file into DataFrame.
read_clipboard : Read text from clipboard into DataFrame.
"""

SEE_ALSO_MARKDOWN = """
#### See Also

- `DataFrame.from_records`: Constructor from tuples, also record arrays.
- `read_table`: Read general delimited file into DataFrame.
- `read_clipboard`: Read text from clipboard into DataFrame.
"""

CODE_MULTI_LINE_CODE_OUTPUT = """
To enforce a single dtype:

>>> df = pd.DataFrame(data=d, dtype=np.int8)
>>> df.dtypes
col1    int8
col2    int8
dtype: object

Constructing DataFrame from numpy ndarray:

>>> df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
...                    columns=['a', 'b', 'c'])
>>> df2
   a  b  c
0  1  2  3
1  4  5  6
2  7  8  9
"""

CODE_MULTI_LINE_CODE_OUTPUT_MARKDOWN = """
To enforce a single dtype:

```python
df = pd.DataFrame(data=d, dtype=np.int8)
df.dtypes
```

```
col1    int8
col2    int8
dtype: object
```


Constructing DataFrame from numpy ndarray:

```python
df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])
df2
```

```
   a  b  c
0  1  2  3
1  4  5  6
2  7  8  9
```

"""

RST_LINK_EXAMPLE = """To learn more about the frequency strings, please see `this link
<https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__."""
RST_LINK_EXAMPLE_MARKDOWN = (
    "To learn more about the frequency strings, please see "
    "[this link](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)."
)
RST_REF_EXAMPLE = """See :ref:`here <timeseries.offset_aliases>` for a list of frequency aliases."""
RST_REF_MARKDOWN = """See here: `timeseries.offset_aliases` for a list of frequency aliases."""

RST_PRODUCTION_LIST_EXAMPLE = """
A function definition defines a user-defined function object:

.. productionlist:: python-grammar
   funcdef: [`decorators`] "def" `funcname` "(" [`parameter_list`] ")"
          : ["->" `expression`] ":" `suite`
   decorators: `decorator`+
   defparameter: `parameter` ["=" `expression`]
   funcname: `identifier`

A function definition is an executable statement.
"""

RST_PRODUCTION_LIST_EXAMPLE_MARKDOWN = """
A function definition defines a user-defined function object:

```python-grammar
funcdef: [`decorators`] "def" `funcname` "(" [`parameter_list`] ")"
       : ["->" `expression`] ":" `suite`
decorators: `decorator`+
defparameter: `parameter` ["=" `expression`]
funcname: `identifier`
```

A function definition is an executable statement.
"""

RST_COLON_CODE_BLOCK = """
For example, the following code ::

   @f1(arg)
   @f2
   def func(): pass

is roughly equivalent to (.. seealso:: exact_conversion) ::

   def func(): pass
   func = f1(arg)(f2(func))

except that the original function is not temporarily bound to the name func.
"""

RST_COLON_CODE_BLOCK_MARKDOWN = """
For example, the following code 

```python
@f1(arg)
@f2
def func(): pass
```

is roughly equivalent to (*See also* exact_conversion) 

```python
def func(): pass
func = f1(arg)(f2(func))
```

except that the original function is not temporarily bound to the name func.
"""

# note: two spaces indent
NUMPY_EXAMPLE = """
The docstring examples assume that `numpy` has been imported as `np`::

  >>> import numpy as np
  
Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1
"""

NUMPY_EXAMPLE_MARKDOWN = """
The docstring examples assume that `numpy` has been imported as `np`

```python
>>> import numpy as np
```

Code snippets are indicated by three greater-than signs

```python
>>> x = 42
>>> x = x + 1
```
"""

NUMPY_MATH_EXAMPLE = """
single-frequency component at linear frequency :math:`f` is
represented by a complex exponential
:math:`a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}`, where :math:`\\Delta t`
is the sampling interval.
"""

NUMPY_MATH_EXAMPLE_MARKDOWN = """
single-frequency component at linear frequency $f$ is
represented by a complex exponential
$a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}$, where $\\Delta t$
is the sampling interval.
"""

PEP_287_CODE_BLOCK = """
Here's a doctest block:

>>> print 'Python-specific usage examples begun with ">>>"'
Python-specific usage examples begun with ">>>"
>>> print '(cut and pasted from interactive sessions)'
(cut and pasted from interactive sessions)"""

PEP_287_CODE_BLOCK_MARKDOWN = """
Here's a doctest block:

```python
print 'Python-specific usage examples begun with ">>>"'
```

```
Python-specific usage examples begun with ">>>"
```

```python
print '(cut and pasted from interactive sessions)'
```

```
(cut and pasted from interactive sessions)
```
"""

RST_HIGHLIGHTED_BLOCK = """
.. highlight:: R

Code block ::

   data.frame()
"""

RST_HIGHLIGHTED_BLOCK_MARKDOWN = """

Code block 

```R
data.frame()
```
"""

NUMPY_NOTE = """
operations and methods.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development.

Some methods will only be available if the corresponding string method is
"""

NUMPY_NOTE_MARKDOWN = """
operations and methods.


---
**Note**

The `chararray` class exists for backwards compatibility with
Numarray, it is not recommended for new development.

---

Some methods will only be available if the corresponding string method is
"""

RST_MATH_EXAMPLE = """
In two dimensions, the DFT is defined as

.. math::
   A_{kl} =  \\\\sum_{m=0}^{M-1} \\\\sum_{n=0}^{N-1}
   a_{mn}\\\\exp\\\\left\\\\{-2\\\\pi i \\\\left({mk\\\\over M}+{nl\\\\over N}\\\\right)\\\\right\\\\}
   \\\\qquad k = 0, \\\\ldots, M-1\\\\quad l = 0, \\\\ldots, N-1,
   
which extends in the obvious way to higher dimensions, and the inverses
"""

RST_MATH_EXAMPLE_MARKDOWN = """
In two dimensions, the DFT is defined as

$$
A_{kl} =  \\\\sum_{m=0}^{M-1} \\\\sum_{n=0}^{N-1}
a_{mn}\\\\exp\\\\left\\\\{-2\\\\pi i \\\\left({mk\\\\over M}+{nl\\\\over N}\\\\right)\\\\right\\\\}
\\\\qquad k = 0, \\\\ldots, M-1\\\\quad l = 0, \\\\ldots, N-1,
$$

which extends in the obvious way to higher dimensions, and the inverses
"""


def test_looks_like_rst_recognises_rst():
  
    assert looks_like_rst(PEP_287_CODE_BLOCK)
    assert looks_like_rst('the following code ::\n\n\tcode')
    assert looks_like_rst('the following code::\n\n\tcode')
    assert looks_like_rst('See Also\n--------\n')


def test_looks_like_rst_ignores_plain_text():

    assert not looks_like_rst('this is plain text')
    assert not looks_like_rst('this might be **markdown**')
    assert not looks_like_rst('::::::\n\n\tcode')
    assert not looks_like_rst('::')
    assert not looks_like_rst('See Also: Interesting Topic')


INTEGRATION = """
Return a fixed frequency DatetimeIndex.

Parameters
----------
start : str or datetime-like, optional
    Frequency strings can have multiples, e.g. '5H'. See
    :ref:`here <timeseries.offset_aliases>` for a list of
    frequency aliases.
tz : str or tzinfo, optional

To learn more about the frequency strings, please see `this link
<https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.
"""


def test_rst_to_markdown_pep287():
    # Converts PEP 287 examples correctly
    # https://www.python.org/dev/peps/pep-0287/
    converted = rst_to_markdown(PEP_287_CODE_BLOCK)
    assert converted == PEP_287_CODE_BLOCK_MARKDOWN


def test_rst_to_markdown_prompt_continuation():

    # handles prompt continuation and multi-line output
    converted = rst_to_markdown(CODE_MULTI_LINE_CODE_OUTPUT)
    assert converted == CODE_MULTI_LINE_CODE_OUTPUT_MARKDOWN

    # converts links
    converted = rst_to_markdown(RST_LINK_EXAMPLE)
    assert converted == RST_LINK_EXAMPLE_MARKDOWN

    converted = rst_to_markdown(INTEGRATION)
    assert RST_LINK_EXAMPLE_MARKDOWN in converted

    # changes highlight
    converted = rst_to_markdown(RST_HIGHLIGHTED_BLOCK)
    assert converted == RST_HIGHLIGHTED_BLOCK_MARKDOWN
  
    # converts production list
    converted = rst_to_markdown(RST_PRODUCTION_LIST_EXAMPLE)
    assert converted == RST_PRODUCTION_LIST_EXAMPLE_MARKDOWN

    # converts inline math
    converted = rst_to_markdown(NUMPY_MATH_EXAMPLE)
    assert converted == NUMPY_MATH_EXAMPLE_MARKDOWN

    # converts math blocks
    converted = rst_to_markdown(RST_MATH_EXAMPLE)
    assert converted == RST_MATH_EXAMPLE_MARKDOWN

    # converts references
    converted = rst_to_markdown(RST_REF_EXAMPLE)
    assert converted == RST_REF_MARKDOWN

    # converts double colon-initiated code block and the preceding lines
    converted = rst_to_markdown(RST_COLON_CODE_BLOCK)
    assert converted == RST_COLON_CODE_BLOCK_MARKDOWN

    # converts double colon-initiated code block with different indent and Python prompt
    converted = rst_to_markdown(NUMPY_EXAMPLE)
    assert converted == NUMPY_EXAMPLE_MARKDOWN

    # converts version changed
    assert rst_to_markdown('.. versionchanged:: 0.23.0') == '*Changed in 0.23.0*'

    # converts "see also" section
    converted = rst_to_markdown(SEE_ALSO)
    assert converted == SEE_ALSO_MARKDOWN
  
    # converts module
    assert (
        rst_to_markdown('Discrete Fourier Transform (:mod:`numpy.fft`)') == 'Discrete Fourier Transform (`numpy.fft`)'
    )

    assert rst_to_markdown(NUMPY_NOTE) == NUMPY_NOTE_MARKDOWN
  


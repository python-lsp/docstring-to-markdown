import pytest

from docstring_to_markdown.rst import looks_like_rst, rst_to_markdown, ReStructuredTextConverter


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

RST_LINK_MULTILINE_EXAMPLE = """
See
`strftime documentation
<https://docs.python.org/3/library/datetime.html
#strftime-and-strptime-behavior>`_ for more.
"""
RST_LINK_MULTILINE_MARKDOWN = """
See
[strftime documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) for more.
"""

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

RST_AUTOSUMMARY_BLOCK = """
Summary

.. autosummary::

   environment.BuildEnvironment
   util.relative_uri
"""


RST_AUTOSUMMARY_BLOCK_MARKDOWN = """
Summary

```
environment.BuildEnvironment
util.relative_uri
```
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
🛈 **Note**

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

MATH_INLINE_BLOCK = """
covariance matrix, `C`, is

.. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

The values of `R` are between -1 and 1, inclusive.
"""

MATH_INLINE_BLOCK_MARKDOWN = """
covariance matrix, `C`, is

$$R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }$$

The values of `R` are between -1 and 1, inclusive.
"""

KWARGS_PARAMETERS = """
Parameters
----------
x : array_like
    Input array.
**kwargs
    For other keyword-only arguments, see the ufunc docs.
"""

KWARGS_PARAMETERS_MARKDOWN = """
#### Parameters

- `x`: array_like
    Input array.
- `**kwargs`:
    For other keyword-only arguments, see the ufunc docs.
"""

NUMPY_ARGS_PARAMETERS = """
Parameters
----------
arys1, arys2, ... : array_like
    One or more input arrays.
"""


NUMPY_ARGS_PARAMETERS_MARKDOWN = """
#### Parameters

- `arys1`, `arys2`, `...`: array_like
    One or more input arrays.
"""


INITIAL_SIGNATURE = """\
absolute1(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Calculate the absolute value element-wise.
"""

INITIAL_SIGNATURE_MARKDOWN = """\
```python
absolute1(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
```

Calculate the absolute value element-wise.
"""


CODE_BLOCK_BUT_NOT_OUTPUT = """
Plot the function over ``[-10, 10]``:

>>> import matplotlib.pyplot as plt

>>> x = np.linspace(start=-10, stop=10, num=101)
>>> plt.plot(x, np.absolute(x))
>>> plt.show()

Plot the function over the complex plane:

>>> xx = x + 1j * x[:, np.newaxis]
>>> plt.imshow(np.abs(xx), extent=[-10, 10, -10, 10], cmap='gray')
>>> plt.show()
"""


CODE_BLOCK_BUT_NOT_OUTPUT_MD = """
Plot the function over ``[-10, 10]``:

```python
import matplotlib.pyplot as plt
```


```python
x = np.linspace(start=-10, stop=10, num=101)
plt.plot(x, np.absolute(x))
plt.show()
```


Plot the function over the complex plane:

```python
xx = x + 1j * x[:, np.newaxis]
plt.imshow(np.abs(xx), extent=[-10, 10, -10, 10], cmap='gray')
plt.show()
```

"""

WARNING_BLOCK = """
Load pickled object from file.

.. warning::
   Loading pickled data received from untrusted sources can be
   unsafe.

Parameters
"""


WARNING_BLOCK_MARKDOWN = """
Load pickled object from file.


---
⚠️  **Warning**

Loading pickled data received from untrusted sources can be
unsafe.

---

Parameters
"""


LINE_WARNING = """
Create a view into the array with the given shape and strides.

.. warning:: This function has to be used with extreme care, see notes.

Parameters
"""

LINE_WARNING_MARKDOWN = """
Create a view into the array with the given shape and strides.

⚠️  **Warning**: This function has to be used with extreme care, see notes.

Parameters
"""


REFERENCES = """
References
----------
.. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
   Biometrika 37, 1-16, 1950.
.. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
   The University of Alberta Press, 1975, pp. 109-110.
.. [3] Wikipedia, "Window function",
   https://en.wikipedia.org/wiki/Window_function
"""

REFERENCES_MARKDOWN = """
#### References

 - [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
   Biometrika 37, 1-16, 1950.
 - [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
   The University of Alberta Press, 1975, pp. 109-110.
 - [3] Wikipedia, "Window function",
   https://en.wikipedia.org/wiki/Window_function
"""


SIMPLE_TABLE = """
.. warning:: This is not a standard simple table

========= ===============================================================
Character Meaning
--------- ---------------------------------------------------------------
'r'       open for reading (default)
'a'       open for writing, appending to the end of the file if it exists
========= ===============================================================
"""

SIMPLE_TABLE_MARKDOWN = """
⚠️  **Warning**: This is not a standard simple table

| Character |                             Meaning                             |
| --------- | --------------------------------------------------------------- |
| 'r'       | open for reading (default)                                      |
| 'a'       | open for writing, appending to the end of the file if it exists |
"""


SIMPLE_TABLE_WITH_MARKUP = """
============================== =======================================
Scalar Type                    Array Type
============================== =======================================
:class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
:class:`bool`                  :class:`pandas.arrays.BooleanArray`
============================== =======================================
"""


SIMPLE_TABLE_WITH_MARKUP_MARKDOWN = """
|    Scalar Type    |           Array Type          |
| ----------------- | ----------------------------- |
| `pandas.Interval` | `pandas.arrays.IntervalArray` |
| `bool`            | `pandas.arrays.BooleanArray`  |
"""


SIMPLE_TABLE_2 = """
.. warning:: This is a standard simple table

=====  =====  =======
  A      B    A and B
=====  =====  =======
False  False  False
True   False  False
=====  =====  =======
"""

SIMPLE_TABLE_2_MARKDOWN = """
⚠️  **Warning**: This is a standard simple table

|   A   |   B   | A and B |
| ----- | ----- | ------- |
| False | False | False   |
| True  | False | False   |
"""

SIMPLE_TABLE_IN_PARAMS = """
Create an array.
Parameters
----------
object : array_like
    An array, any object exposing the array interface, an object whose
    __array__ method returns an array, or any (nested) sequence.
order : {'K', 'A', 'C', 'F'}, optional
    Specify the memory layout of the array.
    If object is an array the following holds.
    ===== ========= ===================================================
    order  no copy                     copy=True
    ===== ========= ===================================================
    'K'   unchanged F & C order preserved, otherwise most similar order
    'F'   F order   F order
    ===== ========= ===================================================
    When ``copy=False`` and a copy is made for other reasons...
subok : bool, optional
    If True, then sub-classes will be passed-through, otherwise
"""

SIMPLE_TABLE_IN_PARAMS_MARKDOWN = r"""
Create an array.
#### Parameters

- `object`: array_like
    An array, any object exposing the array interface, an object whose
    \_\_array\_\_ method returns an array, or any (nested) sequence.
- `order`: {'K', 'A', 'C', 'F'}, optional
    Specify the memory layout of the array.
    If object is an array the following holds.
    | order |  no copy  |                      copy=True                      |
    | ----- | --------- | --------------------------------------------------- |
    | 'K'   | unchanged | F & C order preserved, otherwise most similar order |
    | 'F'   | F order   | F order                                             |
    When ``copy=False`` and a copy is made for other reasons...
- `subok`: bool, optional
    If True, then sub-classes will be passed-through, otherwise
"""

GRID_TABLE_IN_SKLEARN = """
Attributes
----------
cv_results_ : dict of numpy (masked) ndarrays
    A dict with keys as column headers and values as columns, that can be
    imported into a pandas ``DataFrame``.
    For instance the below given table
    +------------+-----------+------------+-----------------+---+---------+
    |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
    +============+===========+============+=================+===+=========+
    |  'poly'    |     --    |      2     |       0.80      |...|    2    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'poly'    |     --    |      3     |       0.70      |...|    4    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
    +------------+-----------+------------+-----------------+---+---------+
    will be represented by a ``cv_results_`` dict
"""

GRID_TABLE_IN_SKLEARN_MARKDOWN = """
#### Attributes

- `cv_results_`: dict of numpy (masked) ndarrays
    A dict with keys as column headers and values as columns, that can be
    imported into a pandas ``DataFrame``.
    For instance the below given table
    | param_kernel | param_gamma | param_degree | split0_test_score | ... | rank_t... |
    | ------------ | ----------- | ------------ | ----------------- | --- | --------- |
    | 'poly'       | --          | 2            | 0.80              | ... | 2         |
    | 'poly'       | --          | 3            | 0.70              | ... | 4         |
    | 'rbf'        | 0.1         | --           | 0.80              | ... | 3         |
    | 'rbf'        | 0.2         | --           | 0.93              | ... | 1         |
    will be represented by a ``cv_results_`` dict
"""


BROKEN_GRID_TABLE = """
+------------+-----------+------------+-----------------+---+---------+
|param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
+============+===========+============+=================+===+=========+
|  'poly'    |     --    |      2     |       0.80      |...|    2    |
+------------+-----------+------------+-----------------+---+---------+
|  'poly'    |     --    |      3     |       0.70      |...|    4    |
someone forgot to close the row above.
"""


BROKEN_GRID_TABLE_MARKDOWN = """
| param_kernel | param_gamma | param_degree | split0_test_score | ... | rank_t... |
| ------------ | ----------- | ------------ | ----------------- | --- | --------- |
| 'poly'       | --          | 2            | 0.80              | ... | 2         |
| 'poly'       | --          | 3            | 0.70              | ... | 4         |
someone forgot to close the row above.
"""


# this format is often used by polars
PARAMETERS_WITHOUT_TYPE = """
Parameters
----------
source
    Path(s) to a file or directory
    When needing to authenticate for scanning cloud locations, see the
    `storage_options` parameter.
columns
    Columns to select. Accepts a list of column indices (starting at zero) or a list
    of column names.
n_rows
    Stop reading from parquet file after reading `n_rows`.
    Only valid when `use_pyarrow=False`.

Returns
-------
DataFrame
"""

PARAMETERS_WITHOUT_TYPE_MARKDOWN = """
#### Parameters

- `source`:
    Path(s) to a file or directory
    When needing to authenticate for scanning cloud locations, see the
    `storage_options` parameter.
- `columns`:
    Columns to select. Accepts a list of column indices (starting at zero) or a list
    of column names.
- `n_rows`:
    Stop reading from parquet file after reading `n_rows`.
    Only valid when `use_pyarrow=False`.

#### Returns

DataFrame
"""

INDENTED_DOCSTRING = """
    Parameters
    ----------
    glob
        Expand path given via globbing rules.
"""

INDENTED_DOCSTRING_MARKDOWN = """
#### Parameters

- `glob`:
    Expand path given via globbing rules.
"""


WARNINGS_IN_PARAMETERS = """
Parameters
----------
glob
    Expand path given via globbing rules.
schema
    Specify the datatypes of the columns. The datatypes must match the
    datatypes in the file(s). If there are extra columns that are not in the
    file(s), consider also enabling `allow_missing_columns`.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
hive_schema
    The column names and data types of the columns by which the data is partitioned.
    If set to `None` (default), the schema of the Hive partitions is inferred.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
try_parse_hive_dates
    Whether to try parsing hive values as date/datetime types.
"""


WARNINGS_IN_PARAMETERS_MARKDOWN = """
#### Parameters

- `glob`:
    Expand path given via globbing rules.
- `schema`:
    Specify the datatypes of the columns. The datatypes must match the
    datatypes in the file(s). If there are extra columns that are not in the
    file(s), consider also enabling `allow_missing_columns`.


    ---
    ⚠️  **Warning**

    This functionality is considered **unstable**. It may be changed
    at any point without it being considered a breaking change.

    ---
- `hive_schema`:
    The column names and data types of the columns by which the data is partitioned.
    If set to `None` (default), the schema of the Hive partitions is inferred.


    ---
    ⚠️  **Warning**

    This functionality is considered **unstable**. It may be changed
    at any point without it being considered a breaking change.

    ---
- `try_parse_hive_dates`:
    Whether to try parsing hive values as date/datetime types.
"""

NESTED_PARAMETERS = """
Parameters
----------
transformers : list of tuples
    List of (name, transformer, columns) tuples.
    name : str
        Like in Pipeline and FeatureUnion, this allows the transformer and
        search.
    transformer : {'drop', 'passthrough'} or estimator
        Estimator must support :term:`fit` and :term:`transform`.
    columns :  str, array-like of str, int, array-like of int, \
            array-like of bool, slice or callable
        Indexes the data on its second axis. Integers are interpreted as
        above. To select multiple columns by name or dtype, you can use
        :obj:`make_column_selector`.
remainder : {'drop', 'passthrough'} or estimator, default='drop'
    By default, only the specified columns in `transformers` are
"""

NESTED_PARAMETERS_MARKDOWN = """
#### Parameters

- `transformers`: list of tuples
    List of (name, transformer, columns) tuples.
    - `name`: str
        Like in Pipeline and FeatureUnion, this allows the transformer and
        search.
    - `transformer`: {'drop', 'passthrough'} or estimator
        Estimator must support `fit` and `transform`.
    - `columns`:  str, array-like of str, int, array-like of int, \
            array-like of bool, slice or callable
        Indexes the data on its second axis. Integers are interpreted as
        above. To select multiple columns by name or dtype, you can use
        `make_column_selector`.
- `remainder`: {'drop', 'passthrough'} or estimator, default='drop'
    By default, only the specified columns in `transformers` are
"""


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


# https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#info-field-lists
SPHINX_SIGNATURE = """
:param str sender: The person sending the message
:param str recipient: The recipient of the message
:param str message_body: The body of the message
:param priority: The priority of the message, can be a number 1-5
:type priority: integer or None
:return: the message id
:rtype: int
:raises ValueError: if the message_body exceeds 160 characters
"""

SPHINX_SIGNATURE_MARKDOWN = """\
- `sender` (`str`): The person sending the message
- `recipient` (`str`): The recipient of the message
- `message_body` (`str`): The body of the message
- `priority` (integer or None): The priority of the message, can be a number 1-5
- returns: the message id
- return type: `int`
- raises `ValueError`: if the message_body exceeds 160 characters
"""

SPHINX_NESTED = """\
.. code-block:: python
    def foo():
        ''':param  str message_body: blah blah'''
"""

SPHINX_NESTED_MARKDOWN = """\
```python
def foo():
    ''':param  str message_body: blah blah'''
```
"""

RST_CASES = {
    'handles prompt continuation and multi-line output': {
        'rst': CODE_MULTI_LINE_CODE_OUTPUT,
        'md': CODE_MULTI_LINE_CODE_OUTPUT_MARKDOWN
    },
    'converts links': {
        'rst': RST_LINK_EXAMPLE,
        'md': RST_LINK_EXAMPLE_MARKDOWN
    },
    'converts multi-line links': {
        'rst': RST_LINK_MULTILINE_EXAMPLE,
        'md': RST_LINK_MULTILINE_MARKDOWN
    },
    'changes highlight': {
        'rst': RST_HIGHLIGHTED_BLOCK,
        'md': RST_HIGHLIGHTED_BLOCK_MARKDOWN
    },
    'converts production list': {
        'rst': RST_PRODUCTION_LIST_EXAMPLE,
        'md': RST_PRODUCTION_LIST_EXAMPLE_MARKDOWN
    },
    'converts inline math': {
        'rst': NUMPY_MATH_EXAMPLE,
        'md': NUMPY_MATH_EXAMPLE_MARKDOWN
    },
    'converts math blocks': {
        'rst': RST_MATH_EXAMPLE,
        'md': RST_MATH_EXAMPLE_MARKDOWN
    },
    'converts inline-block math': {
        'rst': MATH_INLINE_BLOCK,
        'md': MATH_INLINE_BLOCK_MARKDOWN
    },
    'converts refs': {
        'rst': RST_REF_EXAMPLE,
        'md': RST_REF_MARKDOWN
    },
    'converts double colon-initiated code block and the preceding lines': {
        'rst': RST_COLON_CODE_BLOCK,
        'md': RST_COLON_CODE_BLOCK_MARKDOWN
    },
    'converts double colon-initiated code block with different indent and Python prompt': {
        'rst': NUMPY_EXAMPLE,
        'md': NUMPY_EXAMPLE_MARKDOWN
    },
    'converts autosummary block': {
        'rst': RST_AUTOSUMMARY_BLOCK,
        'md': RST_AUTOSUMMARY_BLOCK_MARKDOWN
    },
    'converts version changed': {
        'rst': '.. versionchanged:: 0.23.0',
        'md': '*Changed in 0.23.0*'
    },
    'converts "see also" section': {
        'rst': SEE_ALSO,
        'md': SEE_ALSO_MARKDOWN
    },
    'converts module': {
        'rst': 'Discrete Fourier Transform (:mod:`numpy.fft`)',
        'md': 'Discrete Fourier Transform (`numpy.fft`)'
    },
    'converts note': {
        'rst': NUMPY_NOTE,
        'md': NUMPY_NOTE_MARKDOWN
    },
    'includes kwargs in parameters list': {
        'rst': KWARGS_PARAMETERS,
        'md': KWARGS_PARAMETERS_MARKDOWN
    },
    'converts numpy-style *args parameters': {
        'rst': NUMPY_ARGS_PARAMETERS,
        'md': NUMPY_ARGS_PARAMETERS_MARKDOWN
    },
    'converts signature in the first line': {
        'rst': INITIAL_SIGNATURE,
        'md': INITIAL_SIGNATURE_MARKDOWN
    },
    'separates following paragraph after a code blocks without output': {
        'rst': CODE_BLOCK_BUT_NOT_OUTPUT,
        'md': CODE_BLOCK_BUT_NOT_OUTPUT_MD
    },
    'converts block warnings': {
        'rst': WARNING_BLOCK,
        'md': WARNING_BLOCK_MARKDOWN
    },
    'converts inline-block warnings': {
        'rst': LINE_WARNING,
        'md': LINE_WARNING_MARKDOWN
    },
    'escapes double dunders': {
        # this is guaranteed to not be any rst markup as per
        # https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#inline-markup-recognition-rules
        'rst': '__init__',
        'md': r'\_\_init\_\_'
    },
    'does not escape dunders in code': {
        'rst': '`__init__`',
        'md': '`__init__`'
    },
    'converts bibliographic references': {
        'rst': REFERENCES,
        'md': REFERENCES_MARKDOWN
    },
    'converts sphinx cross-references to func, meth, class, etc.': {
        'rst': ':func:`function1`, :meth:`.Script.inline`, :class:`.Environment`',
        'md': '`function1`, `Script.inline`, `Environment`'
    },
    'converts sphinx cross-references in Python domain': {
        'rst': ':py:func:`function1`, :py:meth:`.Script.inline`, :py:class:`.Environment`',
        'md': '`function1`, `Script.inline`, `Environment`'
    },
    'converts sphinx cross-references in C domain': {
        'rst': ':c:func:`function1`, :c:struct:`Data`',
        'md': '`function1`, `Data`'
    },
    'converts sphinx cross-references in C++ domain': {
        'rst': ':cpp:func:`function1`, :cpp:var:`data`',
        'md': '`function1`, `data`'
    },
    'converts sphinx cross-references in JS domain': {
        'rst': ':js:func:`function1`, :js:class:`Math`',
        'md': '`function1`, `Math`'
    },
    'converts sphinx params': {
        'rst': ':param x: test arg',
        'md': '- `x`: test arg'
    },
    'converts indented sphinx params': {
        'rst': '\t:param x: test arg',
        'md': '- `x`: test arg'
    },
    'converts non-standard simple table': {
        'rst': SIMPLE_TABLE,
        'md': SIMPLE_TABLE_MARKDOWN
    },
    'converts syntax within table': {
        'rst': SIMPLE_TABLE_WITH_MARKUP,
        'md': SIMPLE_TABLE_WITH_MARKUP_MARKDOWN
    },
    'converts standard simple table': {
        'rst': SIMPLE_TABLE_2,
        'md': SIMPLE_TABLE_2_MARKDOWN
    },
    'converts indented simple table': {
        'rst': SIMPLE_TABLE_IN_PARAMS,
        'md': SIMPLE_TABLE_IN_PARAMS_MARKDOWN
    },
    'converts indented grid table': {
        'rst': GRID_TABLE_IN_SKLEARN,
        'md': GRID_TABLE_IN_SKLEARN_MARKDOWN
    },
    'converts broken grid table': {
        'rst': BROKEN_GRID_TABLE,
        'md': BROKEN_GRID_TABLE_MARKDOWN
    },
    'converts nested parameter lists': {
        'rst': NESTED_PARAMETERS,
        'md': NESTED_PARAMETERS_MARKDOWN
    },
    'converts parameter without type': {
        'rst': PARAMETERS_WITHOUT_TYPE,
        'md': PARAMETERS_WITHOUT_TYPE_MARKDOWN
    },
    'converts indented parameters lists': {
        'rst': INDENTED_DOCSTRING,
        'md': INDENTED_DOCSTRING_MARKDOWN
    },
    'converts warnings in parameters lists': {
        'rst': WARNINGS_IN_PARAMETERS,
        'md': WARNINGS_IN_PARAMETERS_MARKDOWN
    },
    'converts sphinx signatures': {
        'rst': SPHINX_SIGNATURE,
        'md': SPHINX_SIGNATURE_MARKDOWN
    },
    'keeps params intact in code blocks': {
        'rst': SPHINX_NESTED,
        'md': SPHINX_NESTED_MARKDOWN
    }
}


def test_looks_like_rst_recognises_rst():
    assert looks_like_rst(PEP_287_CODE_BLOCK)
    assert looks_like_rst('the following code ::\n\n\tcode')
    assert looks_like_rst('the following code::\n\n\tcode')
    assert looks_like_rst('See Also\n--------\n')
    assert looks_like_rst('.. versionadded:: 0.1')
    assert looks_like_rst('Description.\n\n:param spam: eggs.\n')


def test_looks_like_rst_ignores_plain_text():
    assert not looks_like_rst('this is plain text')
    assert not looks_like_rst('this might be **markdown**')
    assert not looks_like_rst('::::::\n\n\tcode')
    assert not looks_like_rst('::')
    assert not looks_like_rst('See Also: Interesting Topic')


def test_rst_to_markdown_pep287():
    # Converts PEP 287 examples correctly
    # https://www.python.org/dev/peps/pep-0287/
    converted = rst_to_markdown(PEP_287_CODE_BLOCK)
    assert converted == PEP_287_CODE_BLOCK_MARKDOWN


def test_integration():
    converted = rst_to_markdown(INTEGRATION)
    assert RST_LINK_EXAMPLE_MARKDOWN in converted


@pytest.mark.parametrize(
    'rst,markdown',
    [[case['rst'], case['md']] for case in RST_CASES.values()],
    ids=RST_CASES.keys()
)
def test_rst_to_markdown(rst, markdown):
    converted = rst_to_markdown(rst)
    print(converted)
    assert converted == markdown


def test_converter():
    converter = ReStructuredTextConverter()
    assert converter.can_convert('.. versionadded:: 0.1')
    assert not converter.can_convert('this is plain text')
    assert converter.convert(PEP_287_CODE_BLOCK) == PEP_287_CODE_BLOCK_MARKDOWN

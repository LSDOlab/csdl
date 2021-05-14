Array Indexing
==============

``omtools`` supports indexing into ``Variable`` objects for explicit
outputs.

In this example, integer indices are used to concatenate multiple
expressions/variables into one variable and extract values from a single
variable representing an array.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_indices_integer.py

Here is the n2 diagram:

.. embed-n2 ::
  ../omtools/examples/valid/ex_indices_integer.py

``omtools`` supports specifying ranges as well as individual indices to
slice and concatenate arrays.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_indices_one_dimensional.py

.. embed-n2 ::
  ../omtools/examples/valid/ex_indices_one_dimensional.py

``omtools`` supports specifying ranges along multiple axes as well as
individual indices and ranges to slice and concatenate arrays.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_indices_multidimensional.py

.. embed-n2 ::
  ../omtools/examples/valid/ex_indices_multidimensional.py

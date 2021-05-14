Array Expansion and Contraction
===============================

This variable can be used to turn a scalar into an array or an array
of lower rank into an array of higher rank.
The values of the smaller array are copied across all values of the
indices in the added dimensions.
Examples for both cases are provided below.

.. autofunction:: omtools.std.expand.expand

.. toctree::
   :maxdepth: 1
   :titlesonly:

   ex_expand_scalar.rst
   ex_expand_array.rst

Simple Explicit Expressions
---------------------------

In this example, we define multiple expressions and register their
outputs.
Note that the outputs need not be registered in the order in which they
are defined, nor do they necessarily appear in the n2 diagram in the
order in which they are registered.
This is due to the fact that ``omtools`` automatically rearranges
expressions so that the ``Component`` objects do not have any
unnecessary feedbacks.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_explicit_binary_operations.py

The ``Component`` objects added to the model are guaranteed to be
connected such that there are no unnecessary feedbacks, regardless of
the order in which each output is defined or registered.
In this case, since there are no cycles, the n2 diagram is upper
triangular.

.. embed-n2::
  ../omtools/examples/valid/ex_explicit_binary_operations.py

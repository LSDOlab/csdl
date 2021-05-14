Expressing Cyclic Relationships with Explicit Outputs
=====================================================

We can also define a value in terms of itself.
In the example below, consider three fixed point iterations.
Each of these fixed point iterations is declared within its own
subsystem so that a solver may be assigned to it.
Note that variables are not rpomoted so that we can use the same name to
refer to different variables, depending on which subsystem they belong
to.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_explicit_cycles.py

.. embed-n2 ::
  ../omtools/examples/valid/ex_explicit_cycles.py

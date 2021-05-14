Failing to Register Outputs Results in No Components
----------------------------------------------------

Here, a subsystem is added as in the previous example, but no outputs
are registered in the parent ``Group``.

This "dead code" does not lead to OpenMDAO constructing any
``Component`` objects, and no ``Component`` objects appear in the n2
diagram for this model, other than the ``Component`` objects that
correspond to the outputs registered in ``'sys'``.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_explicit_no_registered_output.py

.. embed-n2 ::
  ../omtools/examples/valid/ex_explicit_no_registered_output.py

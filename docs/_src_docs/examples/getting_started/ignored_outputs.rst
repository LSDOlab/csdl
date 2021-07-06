Failing to Register Outputs Results in No Components
----------------------------------------------------

Here, a subsystem is added as in the previous example, but no outputs
are registered in the parent ``Model``.

This "dead code" is ignored when the CSDL compiler backend constructs a
computational model.

.. jupyter-execute::
  ../../../../docs/_build/html/examples/ex_explicit_no_registered_output.py
  :linenos:

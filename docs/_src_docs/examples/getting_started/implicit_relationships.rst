Implicit Relationships
======================

It is possible to compute outputs implicitly by defining a residual
variable in terms of the output and inputs.

In the first example, we solve a quadratic equation.
This quadratic has two solutions: ``1`` and ``3``.
Depending on the starting value of the output variable, OpenMDAO will
find one root or the other.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_implicit_apply_nonlinear.py
  :hide-output:

.. jupyter-execute::

  print('using default x=1')
  import omtools.examples.valid.ex_implicit_apply_nonlinear as ex

  print('setting x=1.9')
  ex.prob.set_val('x', 1.9)
  ex.prob.run_model()
  print(ex.prob['x'])

  print('setting x=2.1')
  ex.prob.set_val('x', 2.1)
  ex.prob.run_model()
  print(ex.prob['x'])

The expressions for the residuals will tell OpenMDAO to construct the
relevant ``Component`` objects, but they will be part of a ``Problem``
within the generated ``ImplicitComponent`` object.
As a result, the expressions defined above do not translate to
``Component`` objects in the outer ``Problem`` whose model is displayed
in the n2 diagram below.

.. embed-n2::
  ../omtools/examples/valid/ex_implicit_apply_nonlinear.py

For especially complicated problems, where the residual may converge for
multiple solutions, or where the residual is difficult to converge over
some interval, ``omtools`` provides an API for bracketing solutions.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_implicit_bracketed_scalar.py

Brackets may also be specified for multidimensional array values.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_implicit_bracketed_array.py

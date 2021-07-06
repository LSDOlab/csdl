Implicit Relationships
======================

The ``ImplicitModel`` class provides users with a way to solve for
variables in implicit relationships.
It is possible to compute outputs implicitly by defining a residual
variable in terms of the output and inputs.

In the first example, we solve a quadratic equation.
This quadratic has two solutions: ``1`` and ``3``.
Depending on the starting value of the output variable, CSDL will
find one root or the other.

.. jupyter-execute::
  ../../../../csdl/examples/valid/ex_implicit_apply_nonlinear.py
  :hide-output:

.. jupyter-execute::
  :linenos:

  from csdl.examples.valid.ex_implicit_apply_nonlinear import example
  from csdl_om import Simulator
  sim = example(Simulator)

  print('using default x=1')
  sim.run()
  print('x', sim['x'].shape)
  print(sim['x'])

  print('')
  print('')
  print('setting x=1.9')
  sim['x'] = 1.9
  sim.run()
  print('x', sim['x'].shape)
  print(sim['x'])

  print('')
  print('')
  print('setting x=2.1')
  sim['x'] = 2.1
  sim.run()
  print('x', sim['x'].shape)
  print(sim['x'])

The expressions for the residuals will will be part of a ``Model``
within the generated ``ImplicitModel`` object.

For especially complicated problems, where the residual may converge for
multiple solutions, or where the residual is difficult to converge over
some interval, ``csdl`` provides an API for bracketing solutions.

.. jupyter-execute::
  ../../../../docs/_build/html/examples/ex_implicit_bracketed_scalar.py
  :linenos:

Brackets may also be specified for multidimensional array values.

.. jupyter-execute::
  ../../../../docs/_build/html/examples/ex_implicit_bracketed_array.py
  :linenos:

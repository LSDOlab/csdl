Implicit Relationships with Subsystems
======================================

Residual variables may depend on the result of a subsystem.
For example, the solution to a quadratic equation depends on the
coefficients, but one of those coefficients may not be constant, and may
depend on a subsystem.

In this example, we solve :math:`ax^2+bx+c=0`, but :math:`a` is a fixed
point of :math:`(3 + a - 2a^2)^\frac{1}{4}`.

In order to compute :math:`a`, ``csdl`` creates a new
``openmdao.Problem`` instance within an ``ImplicitModel``.
The ``Problem`` instance contains a model, which is an instance of
``csdl.Model``.
The ``ImplicitModel`` class provides acces to this ``csdl.Model``
object with ``self.model``.
Residuals that depend on subsystems may be defined with
calls to ``self.model.add``.

**Subsystems added by calling ``self.model.add`` are
part of the residual(s), not the ``ImplicitModel``.**

**All inputs to the ``ImplicitModel`` must be declared using calls
to ``self.model.declare_variable`` at the beginning of ``self.setup``,
before any calls to ``self.model.add``.**

In this example, the only input is ``c``.
Both ``a`` and ``b`` are outputs of subsystems used to define the
residual.

.. jupyter-execute::
  ../../../../docs/_build/html/examples/ex_implicit_with_subsystems.py
  :linenos:


Note that calls to ``ImplicitModel.add`` result in adding
a subsystem to the internal ``Problem`` instance (not shown), and not
the ``ImplicitModel`` itself.

Just as with residuals that do not require subsystems to converge,
bracketing solutions is an option as well.

.. jupyter-execute::
  ../../../../docs/_build/html/examples/ex_implicit_with_subsystems_bracketed_scalar.py
  :linenos:

Brackets may also be specified for multidimensional array values.

.. jupyter-execute::
  ../../../../docs/_build/html/examples/ex_implicit_with_subsystems_bracketed_array.py
  :linenos:

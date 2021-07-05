Simple Explicit Expressions with Subsystems
-------------------------------------------

``csdl`` supports constructing model hierarchies via the ``Model.add``
method.

This example starts by creating an input ``'x1'`` to the main model (the
model at the top of the model hierarchy).
The variable ``'x1'`` is created in the current scope, so any parent
model cannot access ``'x1'`` unless ``'x1'`` is promoted to a higher
level in the hierarchy.
A new variable ``y4`` in line 13 is defined in terms of ``x1``.
Note that ``y4`` is the variable's name in Python, not CSDL.
The corresponding name for ``y4`` in CSDL is defined as ``'y4'`` in line
46.
If a variable is not registered with a CSDL name, then CSDL
automatically names the variable.
The Python name for a variable does not need to match its name in CSDL,
but it is good practice to keep them consistent for readability.


In this example, ``'subsystem'`` declares the variable ``'x1'``,
declaring an input to the model ``'subsystem'`` from either a parent
model, or a child model.
In the parent model, the ``create_input`` method creates an input
``'x1'`` to the main model.
The variable ``'x1'`` belongs to the scope of
``ExampleWithSubsystems``.  output in the parent ``Model`` prior to the
call to ``Model.add`` in order to update the input values in
``'sys'``.

Likewise, if the parent ``Model`` is to use an output registered in
``'subsystem'``, such as ``'x2'``, then the user must call
``Model.declare_variable`` after ``Model.add`` for that variable.


.. jupyter-execute::
  ../../../../docs/_build/html/examples/ex_explicit_with_subsystems.py

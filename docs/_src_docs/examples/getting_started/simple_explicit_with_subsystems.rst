Simple Explicit Expressions with Subsystems
-------------------------------------------

In addition to generating ``Component`` objects from expressions,
``omtools`` supports adding subsystems to a ``Group`` class as in
OpenMDAO.
This means that OpenMDAO ``System`` objects, including ``omtools.Model``
objects, can be added to a model built with ``omtools``.

Note that, while ``omtools`` determines execution order of ``Component``
objects, the user is responsible for calling ``Group.add_subsystem``
after the parent ``Group``'s outputs are registered, and before the
parent ``Group``'s inputs are declared.

In this example, ``'subsystem'`` declares ``'x1'`` as an input, so the user
must create an ``'x1'`` output in the parent ``Group`` prior to the call
to ``Group.add_subsystem`` in order to update the input values in
``'sys'``.

Likewise, if the parent ``Group`` is to use an output registered in
``'subsystem'``, such as ``'x2'``, then the user must call
``Group.declare_variable`` after ``Group.add_subsystem`` for that variable.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_explicit_with_subsystems.py

Below is an n2 diagram for a ``Group`` with simple binary expressions
and a subsystem.
The ``Component`` objects added to the model are guaranteed to be
connected such that there are no unnecessary feedbacks, regardless of
the order in which each output is defined or registered.

.. embed-n2::
  ../omtools/examples/valid/ex_explicit_with_subsystems.py

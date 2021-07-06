API
===

Model Classes
-------------

.. autoclass:: csdl.core.model.Model
   :members:

.. autoclass:: csdl.core.implicit_model.ImplicitModel
   :members:

Custom Operation Classes
------------------------

.. autoclass:: csdl.core.custom_operation.CustomOperation
   :members:

The ``ExplicitOperation`` and ``ImplicitOperation`` classes inherit from
the ``CustomOperation`` class.

.. autoclass:: csdl.core.explicit_operation.ExplicitOperation
   :members:

.. autoclass:: csdl.core.implicit_operation.ImplicitOperation
   :members:

Output Classes
--------------


.. autoclass:: csdl.core.explicit_output.ExplicitOutput
  :members: define

.. autoclass:: csdl.core.implicit_output.ImplicitOutput
  :members: define_residual, define_residual_bracketed

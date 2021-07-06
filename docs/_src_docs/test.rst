Testing
=======

``csdl`` provides tests that are independent of the compiler backend
implementation.
That is, tests are implementation-agnostic.

To run all tests for the CSDL compiler, navigate to the directory
containing the ``csdl`` package and run

.. code-block::

    pytest -s --backend <backend>

where ``<backend>`` is the name of the backend implementation, without
angle brackets.
Quotes are not required.

For example, if using ``csdl_om`` as the backend, run

.. code-block::

    pytest -s --backend csdl_om

If all tests pass for one backend implementation, but not another, that
is an indication that the backend implementation that resulted in test
failures has a bug.

If the bug is an indication of lack of test coverage in the frontend,
new tests should be added to ``csdl`` to ensure compliance among all
backend implementations.

Note that this command does not run any test specific to the backend
implementation.

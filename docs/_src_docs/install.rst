Installation Instructions
=========================

To install ``csdl``, first clone the repository and install using `pip`.

.. code-block::

  git clone https://github.com/lsdolab/csdl.git
  pip install -e csdl/

Then install a backend from the list below.

- `csdl_om <https://github.com/lsdolab/csdl_om>`_, which compiles CSDL
  code to an `OpenMDAO <https://openmdao.org>`_ problem.
  No additional knowledge of the OpenMDAO API is required.

Note that this package requires an implementation of the CSDL compiler
backend (options listed above) to run.
Installing ``csdl`` alone will not be enough to run code written using
CSDL.

Packages that implement the compiler backend should depend on ``csdl``.
It is highly recommended to create a different environment for each
backend, as they may require different versions of ``csdl``.

CSDL
====

Introduction
------------

``csdl`` is the compiler frontend for the Computational System Design
Language (CSDL).
CSDL enables users to define and run numerical models for engineering
systems for use in multidisciplinary design optimization problems.

The source code for ``csdl`` is hosted on
`GitHub <https://github.com/lsdolab/csdl>`_, and installation
instructions are located in the README.

Note that this package requires an implementation of the CSDL compiler
backend to run.
Installing `csdl` alone will not be enough to run code written using
CSDL.

Below is a list of backend implementations available for CSDL.

- `csdl_om <https://github.com/lsdolab/csdl_om>`_, which compiles CSDL
  code to an `OpenMDAO <https://openmdao.org>`_ problem.
  No additional knowledge of the OpenMDAO API is required.

Documentation
-------------

To get started using ``csdl``, see the documentation below.

.. toctree::
   :maxdepth: 4
   :titlesonly:


   _src_docs/install.rst
   _src_docs/test.rst
   _src_docs/getting_started.rst
   _src_docs/api/api.rst
   _src_docs/api/std_lib.rst
   _src_docs/api/developer_docs.rst

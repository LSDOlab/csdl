Getting Started
================

A CSDL model is a recipe for building a computational model that can
execute code for model evaluation.
The CSDL compiler frontent ``csdl`` provides the tools necessary to
define a model, and the backend constructs the computational model.
Separating these roles enables users to define their models without the
need to also define derivative computation, which makes it possible to
use the computational model in an optimization setting.

This user guide shows how to define a model using CSDL and how to run
the computational model.
The user guide makes heavy use of examples to help users get started
with CSDL.
The examples in the documentation all use ``csdl_om`` as the CSDL
compiler backend, defined using the statement,
``from csdl_om import Simulator``.
Note that model evaluation is not possible without a CSDL compiler
backend.
To run the examples using a different backend, simply replace
"``csdl_om``" with the compiler backend of your choice.

.. toctree::
   :maxdepth: 3
   :titlesonly:

   examples/getting_started/basic_user_guide.rst

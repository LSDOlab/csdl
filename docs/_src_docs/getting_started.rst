Getting Started
================

The examples below will help get you started with ``omtools``.
``omtools`` provides its own ``Group`` class, which inherits from
OpenMDAO's ``Group`` class.
The ``omtools.Model`` class extends ``openmdao.Group`` so that users can
write expressions using Python syntax that ``omtools`` analyzes to
construct OpenMDAO ``Component`` objects.
The ``Component`` classes required to transform Pythonic expressions
into an OpenMDAO model with predefined derivatives are provided with
``omtools``.
The examples below include ``omtools`` expressions and n2 diagrams to
give an idea of how expressions are transformed to OpenMDAO
``Component`` objects.



.. toctree::
   :maxdepth: 3
   :titlesonly:

   examples/getting_started/basic_user_guide.rst
   examples/getting_started/additional_examples.rst

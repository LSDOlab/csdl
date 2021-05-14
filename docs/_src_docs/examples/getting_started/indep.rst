Independent Variables
=====================

Creating an independent variable always registers that variable as an
output, regardless of whether it is used in an variable, or whether
any variable that uses the independent variable is registered as an
output.

This means that all independent variables are available to parent
``Group`` objects.

In this example, a single independent variable is created within the
model even though there are no dependencies on the independent variable
within the model.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_indep_var_simple.py

.. embed-n2 ::
  ../omtools/examples/valid/ex_indep_var_simple.py

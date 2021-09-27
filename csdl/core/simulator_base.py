from typing import Dict, Any, Tuple
from collections import OrderedDict


class _ReprClass(object):
    """
    Class for defining objects with a simple constant string __repr__.

    This is useful for constants used in arg lists when you want them to appear in
    automatically generated source documentation as a certain string instead of python's
    default representation.
    """
    def __init__(self, repr_string):
        """
        Inititialize the __repr__ string.

        **Parameters**

        repr_string : str
            The string to be returned by __repr__
        """
        self._repr_string = repr_string

    def __repr__(self):
        """
        Return our _repr_string.

        **Returns**

        str
            Whatever string we were initialized with.
        """
        return self._repr_string


# Use this as a special value to be able to tell if the caller set a value for the optional
# out_stream argument. We run into problems running testflo if we use a default of sys.stdout.
_DEFAULT_OUT_STREAM = _ReprClass("DEFAULT_OUT_STREAM")

msg = "The CSDL SimulatorBase class does not implement any methods. Please install a CSDL compiler back end with a Simulator class that conforms to the CSDL Simulator API."


class SimulatorBase:
    """
    A class that can be used as a base class for the ``Simulator`` class
    that a CSDL compiler back end would provide.
    This class is only here so that CSDL users and CSDL compiler back end
    developers have API documentation.
    CSDL users are not to use the ``SimulatorBase`` class provided by
    ``csdl``, only the ``Simulator`` class provided by the CSDL compiler
    back end of choice.
    """
    def __init__(self, model, reorder=False):
        """
        Constructor.
        """
        raise NotImplementedError(msg)

    def __getitem__(self, key):
        """
        Method to get variable values before or after a simulation run
        """
        raise NotImplementedError(msg)

    def __setitem__(self, key, val):
        """
        Method to set values for variables by name
        """
        raise NotImplementedError(msg)

    def run(self):
        """
        Method to run a simulation once. This method should be
        implemented so that it can be called repeatedly to solve an
        optimization problem.
        """
        raise NotImplementedError(msg)

    def compute_total_derivatives(self) -> OrderedDict[str, Any]:
        """
        Method to compute total derivatives for use by an optimizer
        """
        raise NotImplementedError(msg)

    def compute_exact_hessian(self):
        """
        Method to compute exact Hessian
        """
        raise NotImplementedError(msg)

    def check_partials(self):
        """
        Method to compute the error for all partial derivatives of all
        operations within the model.

        **Returns**

        An object that is compatible with ``assert_check_partials``

        """
        raise NotImplementedError(msg)

    def assert_check_partials(self, result, atol=1e-8, rtol=1e-8):
        """
        Method to check that the partial derivatives of all operations
        are within a specified tolerance.

        **Parameters**

        result: Return type of ``check_partials``

        """
        raise NotImplementedError(msg)

    def visualize_implementation(self):
        """
        A method for the back end to provide its own visualization of
        the model.
        """
        raise NotImplementedError(msg)

    def objective(self) -> Dict[str, Any]:
        """
        Method to provide optimizer with objective
        """
        raise NotImplementedError(msg)

    def design_variables(self) -> OrderedDict[str, Dict[str, Any]]:
        """
        Method to provide optimizer with design variables
        """
        raise NotImplementedError(msg)

    def constraints(self) -> OrderedDict[str, Dict[str, Any]]:
        """
        Method to provide optimizer with constraints
        """
        raise NotImplementedError(msg)

    def implicit_outputs(self):
        """
        Method to provide optimizer with implicit_outputs
        """
        raise NotImplementedError(msg)

    def residuals(self):
        """
        Method to provide optimizer with residuals
        """
        raise NotImplementedError(msg)

    def objective_gradient(self) -> OrderedDict[Tuple[str, str], Any]:
        """
        Method to provide optimizer with total derivative of objective
        with respect to design variables
        """
        raise NotImplementedError(msg)

    def constraint_jacobian(self) -> OrderedDict[Tuple[str, str], Any]:
        """
        Method to provide optimizer with total derivatives of
        constraints with respect to design variables
        """
        raise NotImplementedError(msg)

    def residuals_jacobian(self):
        """
        Method to provide optimizer with total derivatives of
        residuals with respect to design variables
        """
        raise NotImplementedError(msg)

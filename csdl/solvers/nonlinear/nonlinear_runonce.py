"""
Define the NonlinearRunOnce class.

This is a simple nonlinear solver that just runs the system once.
"""
from csdl.solvers.nonlinear_solver import NonlinearSolver


class NonlinearRunOnce(NonlinearSolver):
    """
    Simple solver that runs the containing system once.

    This is done without iteration or norm calculation.
    """

    SOLVER = 'NL: RUNONCE'


    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        # Remove unused options from base options here, so that users
        #  attempting to set them will get KeyErrors.
        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_non_converge")

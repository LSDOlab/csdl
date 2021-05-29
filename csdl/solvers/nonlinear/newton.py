"""Define the NewtonSolver class."""

import numpy as np

from csdl.solvers.linesearch.backtracking import BoundsEnforceLS
from csdl.solvers.nonlinear_solver import NonlinearSolver


class NewtonSolver(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    """

    SOLVER = 'NL: Newton'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None

        # Slot for linesearch
        self.linesearch = BoundsEnforceLS()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare(
            'solve_subsystems',
            types=bool,
            desc='Set to True to turn on sub-solvers (Hybrid Newton).')
        self.options.declare('max_sub_solves',
                             types=int,
                             default=10,
                             desc='Maximum number of subsystem solves.')
        self.options.declare(
            'cs_reconverge',
            types=bool,
            default=True,
            desc=
            'When True, when this driver solves under a complex step, nudge '
            'the Solution vector by a small amount so that it reconverges.')
        self.options.declare(
            'reraise_child_analysiserror',
            types=bool,
            default=False,
            desc='When the option is true, a solver will reraise any '
            'AnalysisError that arises during subsolve; when false, it will '
            'continue solving.')

        self.supports['gradients'] = True
        self.supports['implicit_components'] = True

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        super().cleanup()

        if self.linear_solver:
            self.linear_solver.cleanup()
        if self.linesearch:
            self.linesearch.cleanup()

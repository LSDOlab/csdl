"""
Define the BroydenSolver class.
"""

from csdl.solvers.linesearch.backtracking import BoundsEnforceLS
from csdl.solvers.nonlinear_solver import NonlinearSolver


class BroydenSolver(NonlinearSolver):
    """
    Broyden solver.

    Attributes
    ----------
    delta_fxm : ndarray
        Most recent change in residual vector.
    delta_xm : ndarray
        Most recent change in state vector.
    fxm : ndarray
        Most recent residual.
    Gm : ndarray
        Most recent Jacobian matrix.
    linear_solver : LinearSolver
        Linear solver to use for calculating inverse Jacobian.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    size : int
        Total length of the states being solved.
    xm : ndarray
        Most recent state.
    _idx : dict
        Cache of vector indices for each state name.
    _computed_jacobians : int
        Number of computed jacobians.
    _converge_failures : int
        Number of consecutive iterations that failed to converge to the tol definied in options.
    _full_inverse : bool
        When True, Broyden considers the whole vector rather than a list of states.
    _recompute_jacobian : bool
        Flag that becomes True when Broyden detects it needs to recompute the inverse Jacobian.
    """

    SOLVER = 'BROYDEN'

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

        self.size = 0
        self._idx = {}
        self._recompute_jacobian = True
        self.Gm = None
        self.xm = None
        self.fxm = None
        self.delta_xm = None
        self.delta_fxm = None
        self._converge_failures = 0
        self._computed_jacobians = 0

        # This gets set to True if the user doesn't declare any states.
        self._full_inverse = False

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('alpha', default=0.4,
                             desc="Value to scale the starting Jacobian, which is Identity. This "
                                  "option does nothing if you compute the initial Jacobian "
                                  "instead.")
        self.options.declare('compute_jacobian', types=bool, default=True,
                             desc="When True, compute an initial Jacobian, otherwise start "
                                  "with Identity scaled by alpha. Further Jacobians may also be "
                                  "computed depending on the other options.")
        self.options.declare('converge_limit', default=1.0,
                             desc="Ratio of current residual to previous residual above which the "
                                  "convergence is considered a failure. The Jacobian will be "
                                  "regenerated once this condition has been reached a number of "
                                  "consecutive times as specified in max_converge_failures.")
        self.options.declare('cs_reconverge', types=bool, default=True,
                             desc='When True, when this driver solves under a complex step, nudge '
                             'the Solution vector by a small amount so that it reconverges.')
        self.options.declare('diverge_limit', default=2.0,
                             desc="Ratio of current residual to previous residual above which the "
                                  "Jacobian will be immediately regenerated.")
        self.options.declare('max_converge_failures', default=3,
                             desc="The number of convergence failures before regenerating the "
                                  "Jacobian.")
        self.options.declare('max_jacobians', default=10,
                             desc="Maximum number of jacobians to compute.")
        self.options.declare('state_vars', [], desc="List of the state-variable/residuals that "
                                                    "are to be solved here.")
        self.options.declare('update_broyden', default=True,
                             desc="Flag controls whether to perform Broyden update to the "
                                  "Jacobian. There are some applications where it may be useful "
                                  "to turn this off.")
        self.options.declare('reraise_child_analysiserror', types=bool, default=False,
                             desc='When the option is true, a solver will reraise any '
                             'AnalysisError that arises during subsolve; when false, it will '
                             'continue solving.')

        self.supports['gradients'] = True
        self.supports['implicit_components'] = True

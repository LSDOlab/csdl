"""
A few different backtracking line search subsolvers.

BoundsEnforceLS - Only checks bounds and enforces them by one of three methods.
ArmijoGoldsteinLS -- Like above, but terminates with the ArmijoGoldsteinLS condition.

"""

from csdl.solvers.nonlinear_solver import NonlinearSolver


class LinesearchSolver(NonlinearSolver):
    """
    Base class for line search solvers.

    Attributes
    ----------
    _do_subsolve : bool
        Flag used by parent solver to tell the line search whether to solve subsystems while
        backtracking.
    _lower_bounds : ndarray or None
        Lower bounds array.
    _upper_bounds : ndarray or None
        Upper bounds array.
    """
    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        **Parameters**

        **kwargs : dict
            Options dictionary.
        """
        super().__init__(**kwargs)
        # Parent solver sets this to control whether to solve subsystems.
        self._do_subsolve = False
        self._lower_bounds = None
        self._upper_bounds = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt.declare(
            'bound_enforcement',
            default='scalar',
            values=['vector', 'scalar', 'wall'],
            desc=
            "If this is set to 'vector', the entire vector is backtracked together "
            +
            "when a bound is violated. If this is set to 'scalar', only the violating "
            +
            "entries are set to the bound and then the backtracking occurs on the vector "
            +
            "as a whole. If this is set to 'wall', only the violating entries are set "
            +
            "to the bound, and then the backtracking follows the wall - i.e., the "
            + "violating entries do not change during the line search.")
        opt.declare(
            'print_bound_enforce',
            default=False,
            desc=
            "Set to True to print out names and values of variables that are pulled "
            "back to their bounds.")


class BoundsEnforceLS(LinesearchSolver):
    """
    Bounds enforcement only.

    Not so much a linesearch; just check the bounds and if they are violated, then pull back to a
    non-violating point and evaluate.
    """

    SOLVER = 'LS: BCHK'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options

        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        for unused_option in ("atol", "rtol", "maxiter",
                              "err_on_non_converge"):
            opt.undeclare(unused_option)


class ArmijoGoldsteinLS(LinesearchSolver):
    """
    Backtracking line search that terminates using the Armijo-Goldstein condition.

    Attributes
    ----------
    _analysis_error_raised : bool
        Flag is set to True if a subsystem raises an AnalysisError.
    """

    SOLVER = 'LS: AG'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        **Parameters**

        **kwargs : dict
            Options dictionary.
        """
        super().__init__(**kwargs)

        self._analysis_error_raised = False

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt['maxiter'] = 5
        opt.declare(
            'c',
            default=0.1,
            lower=0.0,
            upper=1.0,
            desc="Slope parameter for line of "
            "sufficient decrease. The larger the step, the more decrease is required to "
            "terminate the line search.")
        opt.declare('rho',
                    default=0.5,
                    lower=0.0,
                    upper=1.0,
                    desc="Contraction factor.")
        opt.declare('alpha',
                    default=1.0,
                    lower=0.0,
                    desc="Initial line search step.")
        opt.declare(
            'retry_on_analysis_error',
            default=True,
            desc="Backtrack and retry if an AnalysisError is raised.")
        opt.declare('method',
                    default='Armijo',
                    values=['Armijo', 'Goldstein'],
                    desc="Method to calculate stopping condition.")

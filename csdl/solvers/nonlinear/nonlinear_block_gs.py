"""Define the NonlinearBlockGS class."""

from csdl.solvers.nonlinear_solver import NonlinearSolver


class NonlinearBlockGS(NonlinearSolver):
    """
    Nonlinear block Gauss-Seidel solver.

    Attributes
    ----------
    _delta_outputs_n_1 : ndarray
        Cached change in the full output vector for the previous iteration. Only used if the aitken
        acceleration option is turned on.
    _theta_n_1 : float
        Cached relaxation factor from previous iteration. Only used if the aitken acceleration
        option is turned on.
    """

    SOLVER = 'NL: NLBGS'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        **Parameters**

        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        self._theta_n_1 = 1.0
        self._delta_outputs_n_1 = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare(
            'use_aitken',
            types=bool,
            default=False,
            desc='set to True to use Aitken relaxation')
        self.options.declare(
            'aitken_min_factor',
            default=0.1,
            desc='lower limit for Aitken relaxation factor')
        self.options.declare(
            'aitken_max_factor',
            default=1.5,
            desc='upper limit for Aitken relaxation factor')
        self.options.declare(
            'aitken_initial_factor',
            default=1.0,
            desc='initial value for Aitken relaxation factor')
        self.options.declare(
            'cs_reconverge',
            types=bool,
            default=True,
            desc=
            'When True, when this driver solves under a complex step, nudge '
            'the Solution vector by a small amount so that it reconverges.'
        )
        self.options.declare(
            'use_apply_nonlinear',
            types=bool,
            default=False,
            desc=
            "Set to True to always call apply_nonlinear on the solver's "
            "system after solve_nonlinear has been called.")
        self.options.declare(
            'reraise_child_analysiserror',
            types=bool,
            default=False,
            desc='When the option is true, a solver will reraise any '
            'AnalysisError that arises during subsolve; when false, it will '
            'continue solving.')

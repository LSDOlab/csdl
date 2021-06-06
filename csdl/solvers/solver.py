"""Define the base Solver class."""


from csdl.utils.parameters import Parameters

class Solver(object):
    """
    Base solver class.

    This class is subclassed by NonlinearSolver and LinearSolver,
    which are in turn subclassed by actual solver implementations.

    Attributes
    ----------
    _system : <System>
        Pointer to the owning system.
    _depth : int
        How many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        List of right-hand-side (RHS) vector names.
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _iter_count : int
        Number of iterations for the current invocation of the solver.
    _rec_mgr : <RecordingManager>
        object that manages all recorders added to this solver
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    options : <Parameters>
        Options dictionary.
    recording_options : <Parameters>
        Recording options dictionary.
    supports : <Parameters>
        Options dictionary describing what features are supported by this
        solver.
    _filtered_vars_to_record : Dict
        Dict of list of var names to record
    _norm0 : float
        Normalization factor
    _problem_meta : dict
        Problem level metadata.
    """

    # Object to store some formatting for iprint that is shared across all solvers.
    SOLVER = 'base_solver'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Solver options.
        """
        self._system = None
        self._depth = 0
        self._vec_names = None
        self._mode = 'fwd'
        self._iter_count = 0
        self._problem_meta = None

        # Solver options
        self.options = Parameters()
        self.options.declare('maxiter',
                             types=int,
                             default=10,
                             desc='maximum number of iterations')
        self.options.declare('atol',
                             default=1e-10,
                             desc='absolute error tolerance')
        self.options.declare('rtol',
                             default=1e-10,
                             desc='relative error tolerance')
        self.options.declare('iprint',
                             types=int,
                             default=1,
                             desc='whether to print output')
        self.options.declare(
            'err_on_non_converge',
            types=bool,
            default=False,
            desc="When True, AnalysisError will be raised if we don't converge."
        )

        # Case recording options
        self.recording_options = Parameters()
        self.recording_options.declare(
            'record_abs_error',
            types=bool,
            default=True,
            desc='Set to True to record absolute error at the \
                                       solver level')
        self.recording_options.declare(
            'record_rel_error',
            types=bool,
            default=True,
            desc='Set to True to record relative error at the \
                                       solver level')
        self.recording_options.declare(
            'record_inputs',
            types=bool,
            default=True,
            desc='Set to True to record inputs at the solver level')
        self.recording_options.declare(
            'record_outputs',
            types=bool,
            default=True,
            desc='Set to True to record outputs at the solver level')
        self.recording_options.declare(
            'record_solver_residuals',
            types=bool,
            default=False,
            desc='Set to True to record residuals at the solver level')
        self.recording_options.declare(
            'includes',
            types=list,
            default=['*'],
            desc="Patterns for variables to include in recording. \
                                       Paths are relative to solver's Group. \
                                       Uses fnmatch wildcards")
        self.recording_options.declare(
            'excludes',
            types=list,
            default=[],
            desc="Patterns for vars to exclude in recording. \
                                       (processed post-includes) \
                                       Paths are relative to solver's Group. \
                                       Uses fnmatch wildcards")
        # Case recording related
        self._filtered_vars_to_record = {}
        self._norm0 = 0.0

        # What the solver supports.
        self.supports = Parameters()
        self.supports.declare('gradients', types=bool, default=False)
        self.supports.declare('implicit_components', types=bool, default=False)

        self._declare_options()
        self.options.update(kwargs)

        self.cite = ""

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        if self._system is None:
            return type(self).__name__
        return '{} in {}'.format(type(self).__name__, self._system().msginfo)
    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Solver.
        """
        pass


    def __str__(self):
        """
        Return a string representation of the solver.

        Returns
        -------
        str
            String representation of the solver.
        """
        return self.SOLVER

import os

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.recorders.recording_iteration_stack import Recording

from csdl.solvers.solver import Solver


class LinearSolver(Solver):
    """
    Base class for linear solvers.

    Attributes
    ----------
    _rel_systems : set of str
        Names of systems relevant to the current solve.
    _assembled_jac : AssembledJacobian or None
        If not None, the AssembledJacobian instance used by this solver.
    """
    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        self._rel_systems = None
        self._assembled_jac = None
        super().__init__(**kwargs)

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.options['assemble_jac']:
            yield self

    def add_recorder(self, recorder):
        """
        Add a recorder to the solver's RecordingManager.

        Parameters
        ----------
        recorder : <CaseRecorder>
           A recorder instance to be added to RecManager.
        """
        raise RuntimeError('Recording is not supported on Linear Solvers.')

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            'assemble_jac',
            default=False,
            types=bool,
            desc='Activates use of assembled jacobian by this solver.')

        self.supports.declare('assembled_jac', types=bool, default=True)

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)
        if self.options['assemble_jac'] and not self.supports['assembled_jac']:
            raise RuntimeError("Linear solver %s doesn't support assembled "
                               "jacobians." % self.msginfo)

    def solve(self, vec_names, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        """
        raise NotImplementedError("class %s does not implement solve()." %
                                  (type(self).__name__))

    def _solve(self):
        """
        Run the iterative solver.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)

        while self._iter_count < maxiter and norm > atol and norm / norm0 > rtol:
            with Recording(type(self).__name__, self._iter_count, self) as rec:
                self._single_iteration()
                self._iter_count += 1
                self._run_apply()
                norm = self._iter_get_norm()

                # Save the norm values in the context manager so they can also be recorded.
                rec.abs = norm
                if norm0 == 0:
                    norm0 = 1
                rec.rel = norm / norm0

            self._mpi_print(self._iter_count, norm, norm / norm0)

        system = self._system()

        # flag for the print statements. we only print on root if USE_PROC_FILES is not set to True
        print_flag = system.comm.rank == 0 or os.environ.get('USE_PROC_FILES')

        prefix = self._solver_info.prefix + self.SOLVER

        # Solver terminated early because a Nan in the norm doesn't satisfy the while-loop
        # conditionals.
        if np.isinf(norm) or np.isnan(norm):
            msg = "Solver '{}' on system '{}': residuals contain 'inf' or 'NaN' after {} " + \
                  "iterations."
            if iprint > -1 and print_flag:
                print(
                    prefix +
                    msg.format(self.SOLVER, system.pathname, self._iter_count))

            # Raise AnalysisError if requested.
            if self.options['err_on_non_converge']:
                raise AnalysisError(
                    msg.format(self.SOLVER, system.pathname, self._iter_count))

        # Solver hit maxiter without meeting desired tolerances.
        elif (norm > atol and norm / norm0 > rtol):
            msg = "Solver '{}' on system '{}' failed to converge in {} iterations."

            if iprint > -1 and print_flag:
                print(
                    prefix +
                    msg.format(self.SOLVER, system.pathname, self._iter_count))

            # Raise AnalysisError if requested.
            if self.options['err_on_non_converge']:
                raise AnalysisError(
                    msg.format(self.SOLVER, system.pathname, self._iter_count))

        # Solver converged
        elif iprint == 1 and print_flag:
            print(prefix +
                  ' Converged in {} iterations'.format(self._iter_count))
        elif iprint == 2 and print_flag:
            print(prefix + ' Converged')

    def _run_apply(self):
        """
        Run the apply_linear method on the system.
        """
        self._recording_iter.push(('_run_apply', 0))

        system = self._system()
        scope_out, scope_in = system._get_scope()

        try:
            system._apply_linear(self._assembled_jac, self._vec_names,
                                 self._rel_systems, self._mode, scope_out,
                                 scope_in)
        finally:
            self._recording_iter.pop()

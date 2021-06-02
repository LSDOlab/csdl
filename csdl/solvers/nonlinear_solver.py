from collections import OrderedDict
import os
import pprint
import sys

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.mpi import MPI
from openmdao.warnings import issue_warning, SolverWarning

from csdl.solvers.solver import Solver


class NonlinearSolver(Solver):
    """
    Base class for nonlinear solvers.

    Attributes
    ----------
    _err_cache : dict
        Dictionary holding input and output vectors at start of iteration, if requested.
    """
    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)
        self._err_cache = OrderedDict()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            'debug_print',
            types=bool,
            default=False,
            desc='If true, the values of input and output variables at '
            'the start of iteration are printed and written to a file '
            'after a failure to converge.')
        self.options.declare(
            'stall_limit',
            default=0,
            desc='Number of iterations after which, if the residual norms are '
            'identical within the stall_tol, then terminate as if max '
            'iterations were reached. Default is 0, which disables this '
            'feature.')
        self.options.declare(
            'stall_tol',
            default=1e-12,
            desc='When stall checking is enabled, the threshold below which the '
            'residual norm is considered unchanged.')

    def solve(self):
        """
        Run the solver.
        """
        try:
            self._solve()
        except Exception as err:
            if self.options['debug_print']:
                self._print_exc_debug_info()
            raise err

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        system = self._system()
        if self.options['debug_print']:
            self._err_cache['inputs'] = system._inputs._copy_views()
            self._err_cache['outputs'] = system._outputs._copy_views()

        if self.options['maxiter'] > 0:
            self._run_apply()
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _solve(self):
        """
        Run the iterative solver.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']
        stall_limit = self.options['stall_limit']
        stall_tol = self.options['stall_tol']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)

        stalled = False
        stall_count = 0
        if stall_limit > 0:
            stall_norm = norm0

        while self._iter_count < maxiter and norm > atol and norm / norm0 > rtol and not stalled:
            with Recording(type(self).__name__, self._iter_count, self) as rec:

                if stall_count == 3 and not self.linesearch.options[
                        'print_bound_enforce']:

                    self.linesearch.options['print_bound_enforce'] = True

                    if self._system().pathname:
                        pathname = f"{self._system().pathname}."
                    else:
                        pathname = ""

                    msg = (
                        f"Your model has stalled three times and may be violating the bounds. "
                        f"In the future, turn on print_bound_enforce in your solver options "
                        f"here: \n{pathname}nonlinear_solver.linesearch.options"
                        f"['print_bound_enforce']=True. "
                        f"\nThe bound(s) being violated now are:\n")
                    issue_warning(msg, category=SolverWarning)

                    self._single_iteration()
                    self.linesearch.options['print_bound_enforce'] = False
                else:
                    self._single_iteration()

                self._iter_count += 1
                self._run_apply()
                norm = self._iter_get_norm()

                # Save the norm values in the context manager so they can also be recorded.
                rec.abs = norm
                if norm0 == 0:
                    norm0 = 1
                rec.rel = norm / norm0

                # Check if convergence is stalled.
                if stall_limit > 0:
                    rel_norm = rec.rel
                    norm_diff = np.abs(stall_norm - rel_norm)
                    if norm_diff <= stall_tol:
                        stall_count += 1
                        if stall_count >= stall_limit:
                            stalled = True
                    else:
                        stall_count = 0
                        stall_norm = rel_norm

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
        # Or solver stalled.
        elif (norm > atol and norm / norm0 > rtol) or stalled:

            if stalled:
                msg = "Solver '{}' on system '{}' stalled after {} iterations."
            else:
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
        Run the apply_nonlinear method on the system.
        """
        self._recording_iter.push(('_run_apply', 0))
        try:
            self._system()._apply_nonlinear()
        finally:
            self._recording_iter.pop()

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        return self._system()._residuals.get_norm()

    def _disallow_discrete_outputs(self):
        """
        Raise an exception if any discrete outputs exist in our System.
        """
        if self._system()._var_allprocs_discrete['output']:
            raise RuntimeError(
                "%s has a %s solver and contains discrete outputs %s." %
                (self._system().msginfo, type(self).__name__,
                 sorted(self._system()._var_allprocs_discrete['output'])))

    def _print_exc_debug_info(self):
        coord = self._recording_iter.get_formatted_iteration_coordinate()

        out_strs = [
            "\n# Inputs and outputs at start of iteration '%s':\n" % coord
        ]
        for vec_type, views in self._err_cache.items():
            out_strs.append('\n# nonlinear %s\n' % vec_type)
            out_strs.append(pprint.pformat(views))
            out_strs.append('\n')

        out_str = ''.join(out_strs)
        print(out_str)

        rank = MPI.COMM_WORLD.rank if MPI is not None else 0
        filename = 'solver_errors.%d.out' % rank

        with open(filename, 'a') as f:
            f.write(out_str)
            print("Inputs and outputs at start of iteration have been "
                  "saved to '%s'." % filename)
            sys.stdout.flush()

    def _gs_iter(self):
        """
        Perform a Gauss-Seidel iteration over this Solver's subsystems.
        """
        system = self._system()
        for subsys, _ in system._subsystems_allprocs.values():
            system._transfer('nonlinear', 'fwd', subsys.name)

            if subsys._is_local:
                try:
                    subsys._solve_nonlinear()
                except AnalysisError as err:
                    if 'reraise_child_analysiserror' not in self.options or \
                            self.options['reraise_child_analysiserror']:
                        raise err

from contextlib import contextmanager
from typing import Dict, List, Set, Callable, Union, Tuple

import numpy as np

from csdl.core.variable import Variable
from csdl.core.implicit_output import ImplicitOutput
from csdl.core.model import Model
from csdl.core.input import Input
from csdl.core.output import Output
from csdl.utils.collect_terminals import collect_terminals
from csdl.utils.parameters import Parameters


def _remove_nonresiduals(model):
    """
    Remove dependence of root on expressions that are not
    residuals

    Parameters
    ----------
    var: Variable
        Node that serves as root for DAG
    """
    remove = []
    for var in model.registered_outputs:
        if not isinstance(var, Output):
            # Only Output objects can be residuals
            remove.append(var)
        else:
            # Only Output objects marked as residuals can be residuals
            if var.is_residual is False:
                remove.append(var)
    for rem in remove:
        model.registered_outputs.remove(rem)


def _build_internal_simulator(func: Callable) -> Callable:
    def _build_simulator(self):
        if self._defined is False:
            self._defined = True
            func(self)

            self._model.define()

            # After this runs, the internal model only computes residuals;
            # This is only for performance
            _remove_nonresiduals(self._model)

            # Check if all implicit outputs are defined
            for implicit_output_name, residual in self.out_res_map.items():
                if residual is None:
                    raise ValueError(
                        "{} does not have an associated residual.".format(
                            implicit_output_name))

            # Collect residual expressions and their corresponding inputs
            # and outputs
            for implicit_output in self.res_out_map.values():
                implicit_output_name = implicit_output.name
                residual = self.out_res_map[implicit_output_name]

                # Collect inputs (terminal nodes) for this residual only; no
                # duplicates
                in_vars = list(set(collect_terminals(
                    [],
                    residual,
                    residual,
                )))

                # Store inputs for this implicit output
                self.out_in_map[implicit_output_name] = in_vars

            # Make sure user isn't manually setting input and response
            # variables for internal model
            if len(self._model.constraints) > 0 or len(
                    self._model.design_variables) > 0:
                raise ValueError(
                    "Manually setting input and response variables to the internal model of an ImplicitModel is not supported."
                )

            for implicit_output in self.res_out_map.values():
                implicit_output_name = implicit_output.name
                residual = self.out_res_map[implicit_output_name]

                # set response variables for internal model (residuals)
                self._model.add_constraint(residual)

                # set design variables for internal model (inputs and
                # outputs)
                for in_var in in_vars:
                    in_name = in_var.name
                    if in_name not in self._model.design_variables.keys():
                        self._model.add_design_variable(in_var)

    return _build_simulator


class _ProblemBuilder(type):
    def __new__(cls, name, bases, attr):
        attr['define'] = _build_internal_simulator(attr['define'])
        return super(_ProblemBuilder, cls).__new__(cls, name, bases, attr)


class ImplicitModel(metaclass=_ProblemBuilder):
    """
    Class for creating ``ImplicitModel`` objects that compute
    composite residuals

    Options
    -------
    out_expr: ImplicitOutput
        Object that represents the output of the
        ``CompositeImplicitComp``

    res_expr: Variable
        Object that represents an expression to compute the residual

    """
    _count = -1

    def __init__(self, maxiter=100, visualize=False, **kwargs):
        ImplicitModel._count += 1
        self._defined = False
        self._model = Model()
        self.derivs = dict()
        self.maxiter = 100
        self.visualize = visualize
        self.implicit_outputs = []
        self.res_out_map: Dict[str, ImplicitOutput] = dict()
        self.out_res_map: Dict[str, Union[None, Output]] = dict()
        self.out_in_map: Dict[str, List[Variable]] = dict()
        self.brackets_map: Tuple[Dict[str, np.ndarray],
                                 Dict[str, np.ndarray]] = (dict(), dict())
        self.linear_solver = None
        self._nonlinear_solver = None
        self.parameters = Parameters()
        self.initialize()
        self.parameters.update(kwargs)
        # self.requires_solver = False
        # self.unbracketed_outputs = []
        # self.bracketed_outputs = []
        self.objective = None
        self.constraints = dict()
        self.design_variables = dict()
        self.use_nonlinear_solver = False
        self.solve_using_bracketed_search = False

    def get_nonlinear_solver(self):
        return self._nonlinear_solver

    def set_nonlinear_solver(self, solver):
        if self.solve_using_bracketed_search is True:
            raise ValueError(
                "Cannot set a nonlinear solver when using bracketed search to solve for implicitly defined variables"
            )
        self._nonlinear_solver = solver

    nonlinear_solver = property(get_nonlinear_solver, set_nonlinear_solver)

    def initialize(self):
        """
        User defined method to set parameters
        """
        pass

    def define(self):
        """
        User defined method to define numerical model that defines
        residual computation
        """
        pass

    @contextmanager
    def create_model(self, name: str):
        try:
            m = Model()
            self._model.add(m, name=name, promotes=['*'])
            yield m
        finally:
            # m.define()
            pass

    def create_input(
        self,
        name,
        val=1.0,
        shape=(1, ),
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ) -> Input:
        return self._model.create_input(
            name,
            val=val,
            shape=shape,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            distributed=distributed,
        )

    def connect(self, a: str, b: str):
        self._model.connect(a, b)

    def declare_variable(
        self,
        name: str,
        val=1.0,
        shape=(1, ),
        src_indices=None,
        flat_src_indices=None,
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ) -> Input:
        return self._model.declare_variable(
            name,
            val=val,
            shape=shape,
            src_indices=src_indices,
            flat_src_indices=flat_src_indices,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            distributed=distributed,
        )

    def create_output(
        self,
        name,
        val=1.0,
        shape=(1, ),
        units=None,
        res_units=None,
        desc='',
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ):
        return self._model.create_output(
            name,
            val=val,
            shape=shape,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            res_units=res_units,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
            distributed=distributed,
        )

    def create_implicit_output(
        self,
        name,
        val=1.0,
        shape=(1, ),
        units=None,
        res_units=None,
        desc='',
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ) -> ImplicitOutput:
        """
        Create a value that is computed implicitly

        Parameters
        ----------
        name: str
            Name of variable in CSDL
        shape: Tuple[int]
            Shape of variable
    """
        im = ImplicitOutput(
            self,
            name,
            val=val,
            shape=shape,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            res_units=res_units,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
            distributed=distributed,
        )

        # This is only to check later if ImplicitOutput objects are
        # defined
        self.out_res_map[im.name] = None

        # This is used for visualizing the overall model graph as well
        # as checking if a variable can be added as an objective
        self.implicit_outputs.append(im)

        return im

    def add(
        self,
        submodel,
        name='',
        promotes=None,
        promotes_inputs=None,
        promotes_outputs=None,
    ):
        return self._model.add(
            submodel,
            name=name,
            promotes=promotes,
            promotes_inputs=promotes_inputs,
            promotes_outputs=promotes_outputs,
        )

    def register_output(self, name: str, var: Variable) -> Variable:
        if isinstance(var, ImplicitOutput):
            raise TypeError(
                "Cannot register ImplicitOutput as an output of the internal model that computes residuals"
            )
        return self._model.register_output(name, var)

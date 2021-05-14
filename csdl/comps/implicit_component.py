from typing import Dict, List, Set, Callable

import numpy as np
from openmdao.api import Problem
from openmdao.api import ImplicitComponent as OMImplicitComponent

from csdl.core.subgraph import Subgraph
from csdl.core.variable import Variable
from csdl.core.explicit_output import ExplicitOutput
from csdl.core.implicit_output import ImplicitOutput
from csdl.core.model import Model
from csdl.core.input import Input
from csdl.utils.collect_input_exprs import collect_input_exprs


def _remove_nonresiduals(registered_outputs: List[Variable]):
    """
    Remove dependence of root on expressions that are not
    residuals

    Parameters
    ----------
    root: Variable
        Node that serves as root for DAG
    """
    remove = []
    for expr in registered_outputs:
        if expr.is_residual is False:
            remove.append(expr)
    for rem in remove:
        registered_outputs.remove(rem)


# TODO: disallow independent variables
# TODO: disallow implicit variables in main model class
def _post_setup(func: Callable) -> Callable:
    def _build_problem(self):
        func(self)
        # setup internal problem
        m = self.model

        _remove_nonresiduals(m.registered_outputs)

        # Collect residual expressions and their corresponding inputs
        # and outputs
        for res_expr in m.registered_outputs:
            if isinstance(res_expr, Subgraph) == False and isinstance(
                    res_expr, ExplicitOutput) == False:
                # inputs for this residual only
                in_exprs = set(collect_input_exprs([], res_expr, res_expr))
                # output corresponding to this residual
                out_name = self.model.res_out_map[res_expr.name]

                # accumulate inputs and outputs to set values
                self.all_inputs[out_name] = in_exprs
                self.all_outputs = self.all_outputs.union({out_name})

                # add component output
                # output is an input to internal problem
                self.add_output(
                    out_name,
                    shape=res_expr.shape,
                    val=res_expr.val,
                )

                # add inputs, declare partials (out wrt in)
                for in_expr in in_exprs:
                    if in_expr.name != out_name:
                        self.add_input(
                            in_expr.name,
                            shape=in_expr.shape,
                            val=in_expr.val,
                        )
                        self.declare_partials(
                            of=out_name,
                            wrt=in_expr.name,
                        )

                # residual wrt output
                self.declare_partials(
                    of=out_name,
                    wrt=out_name,
                )

                # set response and design variables to compute derivatives
                # treat inputs and output as inputs to internal model
                # treat residual as output of internal model
                self.prob.model.add_constraint(res_expr.name)
                for in_expr in in_exprs:
                    in_name = in_expr.name
                    if in_name in self.prob.model._design_vars or in_name in self.prob.model._static_design_vars:
                        pass
                    else:
                        self.prob.model.add_design_var(in_name)
                if out_name in self.prob.model._design_vars or out_name in self.prob.model._static_design_vars:
                    pass
                else:
                    self.prob.model.add_design_var(out_name)
        self.prob.setup()

        # set initial values for inputs and output
        for res_expr in self.model.registered_outputs:
            if isinstance(res_expr, Subgraph) == False and isinstance(
                    res_expr, ExplicitOutput) == False:
                out_name = self.model.res_out_map[res_expr.name]
                if len(self.model.out_vals) == 0:
                    self.prob[out_name] = 1
                else:
                    self.prob[out_name] = self.model.out_vals[out_name]
                for in_expr in self.all_inputs[out_name]:
                    self.prob[in_expr.name] = in_expr.val

        # create n2 diagram of internal model for debugging
        if self.n2 == True:
            self.prob.run_model()
            self.prob.model.list_inputs()
            self.prob.model.list_outputs()
            from openmdao.api import n2
            n2(self.prob)

    return _build_problem


class _ProblemBuilder(type):
    def __new__(cls, name, bases, attr):
        attr['setup'] = _post_setup(attr['setup'])
        return super(_ProblemBuilder, cls).__new__(cls, name, bases, attr)


class ImplicitComponent(OMImplicitComponent, metaclass=_ProblemBuilder):
    """
    Class for creating ``ImplicitComponent`` objects that compute
    composite residuals

    Options
    -------
    out_expr: ImplicitOutput
        Object that represents the output of the
        ``CompositeImplicitComp``

    res_expr: Variable
        Object that represents an expression to compute the residual

    """
    def __init__(self, maxiter=100, n2=False, **kwargs):
        super().__init__(**kwargs)
        self._inst_functs = {
            name: getattr(self, name, None)
            for name in
            ['apply_linear', 'apply_multi_linear', 'solve_multi_linear']
        }

        self.prob = Problem()
        self.prob.model = Model()
        self.model = self.prob.model
        self.all_inputs: Dict[Set[Input]] = dict()
        self.all_outputs: Set[str] = set()
        self.derivs = dict()
        self.maxiter = 100
        self.n2 = n2

    def _set_values(self, inputs, outputs):
        for res_expr in self.model.registered_outputs:
            if isinstance(res_expr, Subgraph) == False and isinstance(
                    res_expr, ExplicitOutput) == False:
                out_name = self.model.res_out_map[res_expr.name]
                self.prob[out_name] = outputs[out_name]
                for in_expr in self.all_inputs[out_name]:
                    if in_expr.name != out_name:
                        self.prob[in_expr.name] = inputs[in_expr.name]

    def setup(self):
        pass

    def run(self, inputs, outputs, out_name, bracket):
        self._set_values(inputs, outputs)
        prob = self.prob
        prob[out_name] = bracket
        prob.run_model()

        residuals = dict()
        for res_name, out_name in self.model.res_out_map.items():
            residuals[out_name] = np.array(prob[res_name])

        return residuals

    def apply_nonlinear(self, inputs, outputs, residuals):
        self._set_values(inputs, outputs)
        prob = self.prob
        prob.run_model()

        for res_name, out_name in self.model.res_out_map.items():
            residuals[out_name] = np.array(prob[res_name])

    def solve_nonlinear(self, inputs, outputs):
        for res_expr in self.model.registered_outputs:
            if isinstance(res_expr, Subgraph) == False and isinstance(
                    res_expr, ExplicitOutput) == False:
                out_name = self.model.res_out_map[res_expr.name]
                shape = res_expr.shape

                if self.model.brackets_map is not None:
                    x1 = self.model.brackets_map[0][out_name] * np.ones(shape)
                    x2 = self.model.brackets_map[1][out_name] * np.ones(shape)

                    r1 = self.run(inputs, outputs, out_name, x1)
                    r2 = self.run(inputs, outputs, out_name, x2)
                    mask1 = r1[out_name] >= r2[out_name]
                    mask2 = r1[out_name] < r2[out_name]

                    xp = np.empty(shape)
                    xp[mask1] = x1[mask1]
                    xp[mask2] = x2[mask2]

                    xn = np.empty(shape)
                    xn[mask1] = x2[mask1]
                    xn[mask2] = x1[mask2]

                    for _ in range(self.maxiter):
                        x = 0.5 * xp + 0.5 * xn
                        r = self.run(inputs, outputs, out_name, x)
                        mask_p = r[out_name] >= 0
                        mask_n = r[out_name] < 0
                        xp[mask_p] = x[mask_p]
                        xn[mask_n] = x[mask_n]

                    outputs[out_name] = 0.5 * xp + 0.5 * xn

    def linearize(self, inputs, outputs, jacobian):
        self._set_values(inputs, outputs)
        in_exprs = set()
        for inputs in self.all_inputs.values():
            in_exprs = in_exprs.union(inputs)
        res_names = list(self.model.res_out_map.keys())
        out_names = list(self.all_inputs.keys())

        jac = self.prob.compute_totals(
            of=res_names,
            wrt=[in_expr.name for in_expr in list(in_exprs)] + out_names,
        )

        for res_expr in self.model.registered_outputs:
            if isinstance(res_expr, Subgraph) == False:
                res_name = res_expr.name
                out_name = self.model.res_out_map[res_name]
                for in_expr in self.all_inputs[out_name]:
                    jacobian[out_name, in_expr.name] = jac[res_name,
                                                           in_expr.name]
                jacobian[out_name, out_name] = jac[res_name, out_name]

                self.derivs[out_name] = np.diag(
                    jac[res_name, out_name]).reshape(res_expr.shape)

    def solve_linear(self, d_outputs, d_residuals, mode):
        out_expr = self.options['out_expr']
        out_name = out_expr.name

        for out_name in self.model.res_out_map.values():
            if mode == 'fwd':
                d_outputs[out_name] += 1. / self.derivs[
                    out_name] * d_residuals[out_name]
            else:
                d_residuals[out_name] += 1. / self.derivs[
                    out_name] * d_outputs[out_name]

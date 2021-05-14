from csdl import Model
from csdl.core.variable import Variable
from csdl.core.subsystem import Subsystem
from csdl.core.input import Input
from csdl.core.explicit_output import ExplicitOutput
from openmdao.api import Problem, ExplicitComponent
from typing import Dict, Set, Tuple, Callable
from csdl.utils.collect_input_exprs import collect_input_exprs
import numpy as np
from copy import deepcopy


# TODO: enable ge/le comparisons
# TODO: support multiple outputs from expr_true, expr_false
class ConditionalComponent(ExplicitComponent):
    def initialize(self):
        self.options.declare('out_name', types=str)
        self.options.declare('condition', types=Variable)
        self.options.declare('expr_true', types=Variable)
        self.options.declare('expr_false', types=Variable)
        self.options.declare('n2', types=bool, default=False)
        self.condition = Problem()
        self.condition.model = Model()
        self.ptrue = Problem()
        self.ptrue.model = Model()
        self.pfalse = Problem()
        self.pfalse.model = Model()

        # self.all_outputs: Set[str] = set()
        self.in_exprs_condition: Dict[Set[Input]] = dict()
        self.in_exprs_true: Dict[Set[Input]] = dict()
        self.in_exprs_false: Dict[Set[Input]] = dict()

        self.derivs = dict()

    # TODO: allow condition to be an input
    # TODO: check number of expressions for expr_true, expr_false
    def setup(self):
        out_name = self.options['out_name']
        condition = self.options['condition']
        if isinstance(condition, Input):
            raise TypeError("Condition must not be an Input object")
        expr_true = self.options['expr_true']
        if isinstance(expr_true, Input):
            raise TypeError(
                "Variable object to evaluate when condition is TRUE must not be an Input object"
            )
        expr_false = self.options['expr_false']
        if isinstance(expr_false, Input):
            raise TypeError(
                "Variable object to evaluate when condition is FALSE must not be an Input object"
            )

        if expr_true.shape != expr_false.shape:
            raise ValueError(
                "Variable shapes must be the same for Variable objects for both branches of execution"
            )

        self.add_output(
            out_name,
            shape=expr_true.shape,
            val=expr_true.val,
        )

        # collect input expressions for all three problems
        self.in_exprs_condition = set(
            collect_input_exprs([], condition, condition))
        self.in_exprs_true = set(collect_input_exprs([], expr_true, expr_true))
        self.in_exprs_false = set(
            collect_input_exprs([], expr_false, expr_false))

        # TODO: enable multiple outputs
        # self.all_inputs[out_name] = in_exprs
        # self.all_outputs = self.all_outputs.union({out_name})

        # add inputs, declare partials (out wrt in)
        for in_expr in set(self.in_exprs_condition, self.in_exprs_true,
                           self.in_exprs_false):
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

        # setup response variables
        self.condition.model.add_constraint(condition.name)
        self.ptrue.model.add_constraint(expr_true.name)
        self.pfalse.model.add_constraint(expr_false.name)

        # setup input variables
        for in_expr in self.in_exprs_condition:
            in_name = in_expr.name
            if in_name in self.condition.model._design_vars or in_name in self.condition.model._static_design_vars:
                pass
            else:
                self.condition.model.add_design_var(in_name)
        for in_expr in self.in_exprs_true:
            in_name = in_expr.name
            if in_name in self.ptrue.model._design_vars or in_name in self.ptrue.model._static_design_vars:
                pass
            else:
                self.ptrue.model.add_design_var(in_name)
        for in_expr in self.in_exprs_false:
            in_name = in_expr.name
            if in_name in self.pfalse.model._design_vars or in_name in self.pfalse.model._static_design_vars:
                pass
            else:
                self.pfalse.model.add_design_var(in_name)

        # setup internal problems
        self.condition.model._root = deepcopy(condition)
        self.condition.setup()
        self.ptrue.model._root = deepcopy(expr_true)
        self.ptrue.setup()
        self.pfalse.model._root = deepcopy(expr_false)
        self.pfalse.setup()

        # create n2 diagram of internal model for debugging
        if self.options['n2'] == True:
            self.ptrue.run_model()
            self.ptrue.model.list_inputs()
            self.ptrue.model.list_outputs()
            self.pfalse.run_model()
            self.pfalse.model.list_inputs()
            self.pfalse.model.list_outputs()
            from openmdao.api import n2
            # TODO: check that two n2 diagrams are created, make sure
            # both show up in docs
            n2(self.ptrue)
            n2(self.pfalse)

    def _set_values(self, inputs):
        for in_expr in self.in_exprs_condition:
            self.condition[in_expr.name] = inputs[in_expr.name]
        for in_expr in self.in_exprs_true:
            self.ptrue[in_expr.name] = inputs[in_expr.name]
        for in_expr in self.in_exprs_false:
            self.pfalse[in_expr.name] = inputs[in_expr.name]

    def compute(self, inputs, outputs):
        out_name = self.options['out_name']
        condition = self.options['condition']

        self.condition.run_model()
        prob = None
        if self.condition[condition.name] > 0:
            prob = self.ptrue
        else:
            prob = self.pfalse
        prob.run_model()
        outputs[out_name] = np.array(prob[out_name])

    def compute_partials(self, inputs, partials):
        out_name = self.options['out_name']
        condition = self.options['condition']
        expr_true = self.options['expr_true']
        expr_false = self.options['expr_false']

        self._set_values(inputs)
        prob = None
        in_exprs = set()
        branch_expr_name = ''
        if self.condition[condition.name] > 0:
            prob = self.ptrue
            in_exprs = self.in_exprs_true
            branch_expr_name = expr_true.name
        else:
            prob = self.pfalse
            in_exprs = self.in_exprs_false
            branch_expr_name = expr_false.name
        jac = prob.compute_totals(
            of=branch_expr_name,
            wrt=[in_expr.name
                 for in_expr in list(in_exprs)] + branch_expr_name,
        )
        for in_expr in in_exprs:
            partials[out_name, in_expr.name] = jac[branch_expr_name,
                                                   in_expr.name]

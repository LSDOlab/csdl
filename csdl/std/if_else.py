# from csdl.lang.variable import Variable
# # from csdl.comps.conditional_component import ConditionalComponent
# from openmdao.api import ExplicitComponent
# import numpy as np

# class ConditionalComponent(ExplicitComponent):
#     def initialize(self):
#         self.options.declare('out_name')
#         self.options.declare('condition')
#         self.options.declare('expr_true')
#         self.options.declare('expr_false')

#     def setup(self):
#         out_name = self.options['out_name']
#         condition = self.options['condition']
#         expr_true = self.options['expr_true']
#         expr_false = self.options['expr_false']

#         self.add_input(condition.name, condition.val)
#         self.add_input(expr_true.name, expr_true.val)
#         self.add_input(expr_false.name, expr_false.val)
#         self.add_output(out_name)

#         self.declare_partials(
#             of=out_name,
#             wrt=[expr_true.name, expr_false.name],
#             rows=np.arange(np.prod(expr_true.shape)),
#             cols=np.arange(np.prod(expr_true.shape)),
#         )

#     def compute(self, inputs, outputs):
#         out_name = self.options['out_name']
#         condition = self.options['condition']
#         expr_true = self.options['expr_true']
#         expr_false = self.options['expr_false']

#         if inputs[condition.name] < 0:
#             outputs[out_name] = inputs[expr_false.name]
#         else:
#             outputs[out_name] = inputs[expr_true.name]

#     def compute_partials(self, inputs, partials):
#         out_name = self.options['out_name']
#         condition = self.options['condition']
#         expr_true = self.options['expr_true']
#         expr_false = self.options['expr_false']

#         if inputs[condition.name] < 0:
#             partials[out_name, expr_false.name] = inputs[expr_false.name]
#             partials[out_name, expr_true.name] = 0
#         else:
#             partials[out_name, expr_true.name] = inputs[expr_true.name]
#             partials[out_name, expr_false.name] = 0

# def if_else(
#     condition: Variable,
#     expr_true: Variable,
#     expr_false: Variable,
# ):
#     if expr_true.shape != expr_false.shape:
#         raise ValueError(
#             "Variable shapes must be the same for Variable objects for both branches of execution"
#         )

#     out = Variable()
#     out.add_dependency_node(condition)
#     out.add_dependency_node(expr_true)
#     out.add_dependency_node(expr_false)
#     out.build = lambda: ConditionalComponent(
#         out_name=out.name,
#         condition=condition,
#         expr_true=expr_true,
#         expr_false=expr_false,
#     )
#     return out

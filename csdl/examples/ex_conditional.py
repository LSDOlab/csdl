from csdl import Model, ImplicitModel
import csdl
import numpy as np

# class ExampleLiterals(Model):
#     def define(self):
# x = self.create_input('x', shape=(3, 4))
# y = self.declare_variable('y', val=4)
# z = csdl.if_else(
#     x,
#     2 * x,
#     2 * y,
# )

# x = self.create_input('x', val=3)
# y = self.declare_variable('y', val=4)
# z, _ = csdl.if_else(
#     x,
#     (x, y),
#     (x, y),
# )

# x = self.declare_variable('x', val=3)
# y = self.declare_variable('y', val=4)
# z = csdl.if_else(
#     2 * x - 8,
#     2 * x,
#     3 * y,
# )
# self.register_output('z', z)

# prob = Problem()
# prob.model = ExampleLiterals()
# prob.setup(force_alloc_complex=True)
# prob.run_model()

# print('z', prob['z'].shape)
# print(prob['z'])

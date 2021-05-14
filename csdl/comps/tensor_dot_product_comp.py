import numpy as np
from openmdao.api import ExplicitComponent


# Note: This component uses dense partials, use einsum_partials(partial_format = 'sparse') if partials are large and sparse
class TensorDotProductComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_names',
                             default=None,
                             types=list,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shape', types=tuple)
        self.options.declare('axis', types=int)
        self.options.declare('out_shape', default=None, types=tuple)
        self.options.declare('in_vals', types=list)

    def setup(self):
        in_names = self.options['in_names']
        in_shape = self.options['in_shape']
        out_name = self.options['out_name']
        out_shape = self.options['out_shape']
        axis = self.options['axis']
        in_vals = self.options['in_vals']

        self.add_input(in_names[0], shape=in_shape, val=in_vals[0])
        self.add_input(in_names[1], shape=in_shape, val=in_vals[1])

        if out_shape == None:
            out_shape = tuple(np.delete(list(in_shape), axis))

        self.add_output(out_name, shape=out_shape)

        # This is all for partial derivative computation
        in_size = np.prod(in_shape)
        out_size = np.prod(out_shape)

        cols = np.arange(in_size)

        rows = np.unravel_index(np.arange(in_size), shape=in_shape)
        rows = np.delete(np.array(rows), axis, axis=0)
        rows = np.ravel_multi_index(rows, dims=out_shape)

        self.declare_partials(out_name, in_names[0], rows=rows, cols=cols)
        self.declare_partials(out_name, in_names[1], rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        axis = self.options['axis']

        outputs[out_name] = np.sum(inputs[in_names[0]] * inputs[in_names[1]],
                                   axis=axis)

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        axis = self.options['axis']

        partials[out_name, in_names[0]] = inputs[in_names[1]].flatten()
        partials[out_name, in_names[1]] = inputs[in_names[0]].flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import time

    prob = Problem()

    comp = IndepVarComp()
    shape = (2, 3, 4)
    size = np.prod(shape)
    comp.add_output('x', np.arange(size).reshape(shape))
    comp.add_output('y', np.arange(size, 2 * size).reshape(shape))

    prob.model.add_subsystem('input_comp', comp, promotes=['*'])

    comp = TensorDotProductComp(
        in_names=['x', 'y'],
        in_shape=shape,
        axis=1,
        out_name='f',
    )

    prob.model.add_subsystem('comp', comp, promotes=['*'])

    start = time.time()

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    end = time.time()

    print(end - start)

# import numpy as np
# from openmdao.api import ExplicitComponent

# # Note: This component uses dense partials, use einsum_partials(partial_format = 'sparse') if partials are large and sparse
# class TensorInnerProductComp(ExplicitComponent):
#     def initialize(self):
#         self.options.declare('in_names',
#                              default=None,
#                              types=list,
#                              allow_none=True)
#         self.options.declare('out_name', types=str)
#         self.options.declare('in_shape', types=tuple)
#         self.options.declare('axis', types=int)
#         self.options.declare('out_shape', default=None, types=tuple)
#         self.options.declare('in_vals', types=list)

#     def setup(self):
#         in_names = self.options['in_names']
#         in_shape = self.options['in_shape']
#         out_name = self.options['out_name']
#         out_shape = self.options['out_shape']
#         axis = self.options['axis']
#         in_vals = self.options['in_vals']

#         self.add_input(in_names[0], shape=in_shape, val=in_vals[0])
#         self.add_input(in_names[1], shape=in_shape, val=in_vals[1])

#         if out_shape == None:
#             out_shape = tuple(np.delete(list(in_shape), axis))

#         self.add_output(out_name, shape=out_shape)

#         # This is all for partial derivative computation

#         in_size0 = np.prod(in_shape)
#         in_size1 = np.prod(in_shape)

#         flat_indices0 = np.arange(in_size0)
#         flat_indices1 = np.arange(in_size1)

#         ind0 = np.unravel_index(flat_indices0, in_shape)
#         ind1 = np.unravel_index(flat_indices1, in_shape)

#         I0_ind = tuple(2 * list(ind0))
#         I1_ind = tuple(2 * list(ind1))

#         self.I0 = np.zeros(tuple(2 * list(in_shape)))
#         self.I1 = np.zeros(tuple(2 * list(in_shape)))

#         self.I0[I0_ind] += 1
#         self.I1[I1_ind] += 1

#         # Compute new axes locations of the partial matrix wrt first input
#         num_total_axes0 = len(in_shape)
#         num_total_axes1 = len(in_shape)
#         num_rem_axes0 = len(new_in0_shape)
#         num_partial_axes0 = num_rem_axes0 + num_total_axes0
#         total_num_partial_axes0 = num_rem_axes0 + num_rem_axes0 + num_total_axes1

#         self.new_axes_locations = list(np.arange(num_rem_axes0)) + list(
#             np.arange(num_partial_axes0, total_num_partial_axes0)) + list(
#                 np.arange(num_rem_axes0, num_partial_axes0))

#         self.declare_partials(out_name, in_names[0])
#         self.declare_partials(out_name, in_names[1])

#     def compute(self, inputs, outputs):
#         in_names = self.options['in_names']
#         out_name = self.options['out_name']
#         axis = self.options['axis']

#         outputs[out_name] = np.sum(inputs[in_names[0]] * inputs[in_names[1]],
#                                    axis=axis)

#     def compute_partials(self, inputs, partials):
#         in_names = self.options['in_names']
#         out_name = self.options['out_name']
#         axis = self.options['axis']

#         temp_partials = np.tensordot(self.I0, inputs[in_names[1]], axes=axes)

#         partials[out_name, in_names[0]] = np.transpose(temp_partials,
#                                                        self.new_axes_locations)
#         partials[out_name, in_names[1]] = np.tensordot(inputs[in_names[0]],
#                                                        self.I1,
#                                                        axes=axis)

# if __name__ == '__main__':
#     from openmdao.api import Problem, IndepVarComp
#     import time

#     prob = Problem()

#     comp = IndepVarComp()
#     shape = (2, 3, 4)
#     comp.add_output('x', np.arange(np.prod(shape)).reshape(shape))
#     comp.add_output('y', np.arange(np.prod(shape)).reshape(shape))

#     prob.model.add_subsystem('input_comp', comp, promotes=['*'])

#     comp = TensorInnerProductComp(
#         in_names=['x', 'y'],
#         in_shape=shape,
#         axis=1,
#         out_name='f',
#     )

#     prob.model.add_subsystem('comp', comp, promotes=['*'])

#     start = time.time()

#     prob.setup(check=True)
#     prob.run_model()
#     prob.check_partials(compact_print=True)

#     end = time.time()

#     print(end - start)

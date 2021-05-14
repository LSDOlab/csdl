import numpy as np
from openmdao.api import ExplicitComponent


class TensorOuterProductComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_names',
                             default=None,
                             types=list,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shapes', types=list)
        self.options.declare('in_vals', types=list)

    def setup(self):
        in_names = self.options['in_names']
        in_shapes = self.options['in_shapes']
        out_name = self.options['out_name']
        in_vals = self.options['in_vals']

        alphabets = 'abcdefghijklmnopqrstuvwxyz'
        rank0 = len(in_shapes[0])
        rank1 = len(in_shapes[1])
        out_rank = rank0 + rank1
        in0 = alphabets[:rank0]
        in1 = alphabets[rank0:out_rank]
        out = alphabets[:out_rank]
        self.subscript = '{},{}->{}'.format(in0, in1, out)

        self.add_input(in_names[0], shape=in_shapes[0], val=in_vals[0])
        self.add_input(in_names[1], shape=in_shapes[1], val=in_vals[1])

        self.out_shape = tuple(list(in_shapes[0]) + list(in_shapes[1]))
        self.add_output(out_name, shape=self.out_shape)

        out_size = np.prod(self.out_shape)
        self.in_size0 = np.prod(in_shapes[0])
        self.in_size1 = np.prod(in_shapes[1])

        rows = np.arange(out_size)
        cols0 = np.repeat(np.arange(self.in_size0), self.in_size1)
        cols1 = np.tile(np.arange(self.in_size1), self.in_size0)

        self.declare_partials(out_name, in_names[0], rows=rows, cols=cols0)
        self.declare_partials(out_name, in_names[1], rows=rows, cols=cols1)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']

        outputs[out_name] = np.einsum(self.subscript, inputs[in_names[0]],
                                      inputs[in_names[1]])

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        in_shapes = self.options['in_shapes']
        out_name = self.options['out_name']

        partials[out_name,
                 in_names[0]] = np.tile(inputs[in_names[1]].flatten(),
                                        self.in_size0)
        partials[out_name,
                 in_names[1]] = np.repeat(inputs[in_names[0]].flatten(),
                                          self.in_size1)


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import time

    prob = Problem()

    comp = IndepVarComp()
    shape1 = (2, 3, 4)
    shape2 = (7, 2)
    comp.add_output('x', np.arange(np.prod(shape1)).reshape(shape1))
    comp.add_output('y', np.arange(np.prod(shape2)).reshape(shape2))

    prob.model.add_subsystem('input_comp', comp, promotes=['*'])

    comp = TensorOuterProductComp(
        in_names=['x', 'y'],
        in_shapes=[shape1, shape2],
        out_name='f',
    )

    prob.model.add_subsystem('comp', comp, promotes=['*'])

    start = time.time()

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    end = time.time()

    print(end - start)

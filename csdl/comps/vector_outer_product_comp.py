import numpy as np
from openmdao.api import ExplicitComponent


class VectorOuterProductComp(ExplicitComponent):
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

        # print(in_shapes)

        self.add_input(in_names[0], shape=in_shapes[0], val=in_vals[0])
        self.add_input(in_names[1], shape=in_shapes[1], val=in_vals[1])

        self.add_output(out_name, shape=(in_shapes[0], in_shapes[1]))

        rows = np.arange(in_shapes[0] * in_shapes[1])
        cols0 = np.repeat(np.arange(in_shapes[0]), in_shapes[1])
        cols1 = np.tile(np.arange(in_shapes[1]), in_shapes[0])

        self.declare_partials(out_name, in_names[0], rows=rows, cols=cols0)
        self.declare_partials(out_name, in_names[1], rows=rows, cols=cols1)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']

        outputs[out_name] = np.outer(inputs[in_names[0]], inputs[in_names[1]])

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        in_shapes = self.options['in_shapes']
        out_name = self.options['out_name']

        partials[out_name, in_names[0]] = np.tile(inputs[in_names[1]],
                                                  in_shapes[0])
        partials[out_name, in_names[1]] = np.repeat(inputs[in_names[0]],
                                                    in_shapes[1])


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import time

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.arange(7))
    comp.add_output('y', np.arange(3, 6))

    prob.model.add_subsystem('input_comp', comp, promotes=['*'])

    comp = VectorOuterProductComp(
        in_names=['x', 'y'],
        in_shapes=[7, 3],
        out_name='f',
    )

    prob.model.add_subsystem('comp', comp, promotes=['*'])

    start = time.time()

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    end = time.time()

    print(end - start)

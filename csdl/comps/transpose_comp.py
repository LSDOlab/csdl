import numpy as np
from openmdao.api import ExplicitComponent


class TransposeComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_name',
                             default=None,
                             types=str,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shape', types=tuple)
        self.options.declare('out_shape', types=tuple, default=None)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        in_shape = self.options['in_shape']
        out_name = self.options['out_name']
        out_shape = self.options['out_shape']
        val = self.options['val']
        self.rank = len(in_shape)

        self.add_input(in_name, shape=in_shape, val=val)

        if out_shape == None:
            out_shape = in_shape[::-1]

        self.add_output(out_name, shape=out_shape)

        size = np.prod(in_shape)

        rows = np.arange(size)
        initial_locations = np.arange(size).reshape(in_shape)
        new_locations = np.transpose(initial_locations)
        cols = new_locations.flatten()

        # Alternate method
        # ================

        # cols = np.arange(size)
        # initial_locations = np.unravel_index(np.arange(size), shape = in_shape)
        # new_locations = np.array(initial_locations)[::-1, :]
        # rows = np.ravel_multi_index(new_locations, dims = out_shape)

        val = np.ones((size, ))
        self.declare_partials(out_name, in_name, rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.transpose(inputs[in_name])


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import time

    shape = (2, 7, 4)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape))

    prob.model.add_subsystem('input_comp', comp, promotes=['*'])

    comp = TransposeComp(
        in_name='x',
        in_shape=(2, 7, 4),
        out_name='f',
    )

    prob.model.add_subsystem('comp', comp, promotes=['*'])

    start = time.time()

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    end = time.time()

    print(end - start)

import numpy as np
from openmdao.api import ExplicitComponent
from csdl.utils.reorder_axes_utils import compute_new_axes_locations


class ReorderAxesComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_name',
                             default=None,
                             types=str,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shape', types=tuple)
        self.options.declare('operation', types=str)
        self.options.declare('out_shape', types=tuple, default=None)
        self.options.declare('new_axes_locations', types=list, default=None)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        in_shape = self.options['in_shape']
        out_name = self.options['out_name']
        out_shape = self.options['out_shape']
        operation = self.options['operation']
        new_axes_locations = self.options['new_axes_locations']
        val = self.options['val']

        self.add_input(in_name, shape=in_shape, val=val)

        if new_axes_locations == None:
            self.new_axes_locations = compute_new_axes_locations(
                in_shape, operation)
        else:
            self.new_axes_locations = new_axes_locations

        if out_shape == None:
            out_shape = tuple(in_shape[i] for i in self.new_axes_locations)

        self.add_output(out_name, shape=out_shape)

        size = np.prod(in_shape)
        rows = np.arange(size)

        initial_locations = np.arange(size).reshape(in_shape)
        new_locations = np.transpose(initial_locations,
                                     self.new_axes_locations)
        cols = new_locations.flatten()

        val = np.ones((size, ))

        self.declare_partials(out_name, in_name, rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.transpose(inputs[in_name],
                                         self.new_axes_locations)

        # Alternate solution:
        # ==================
        # outputs[out_name] = np.moveaxis(inputs[in_name], [source1, source2], [destination1, destination2])


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, 7, 4)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape))

    prob.model.add_subsystem('input_comp', comp, promotes=['*'])

    comp = ReorderAxesComp(in_name='x',
                           in_shape=(2, 7, 4),
                           out_name='f',
                           operation='ijk->kij')

    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

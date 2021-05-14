import numpy as np

from openmdao.api import ExplicitComponent

from csdl.utils.get_array_indices import get_array_indices

alphabet = 'abcdefghij'


class CrossProductComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('axis', types=int)
        self.options.declare('in1_name', types=str)
        self.options.declare('in2_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('in1_val', types=np.ndarray)
        self.options.declare('in2_val', types=np.ndarray)

    def setup(self):
        shape = self.options['shape']
        axis = self.options['axis']
        in1_name = self.options['in1_name']
        in2_name = self.options['in2_name']
        out_name = self.options['out_name']
        in1_val = self.options['in1_val']
        in2_val = self.options['in2_val']

        self.add_input(in1_name, shape=shape, val=in1_val)
        self.add_input(in2_name, shape=shape, val=in2_val)
        self.add_output(out_name, shape=shape)

        indices = get_array_indices(*shape)

        self.shape_without_axis = shape[:axis] + shape[axis + 1:]

        ones = np.ones(3, int)

        rank = len(self.shape_without_axis)

        einsum_string_rows = '{}y{},z->{}{}yz'.format(
            alphabet[:axis],
            alphabet[axis:rank],
            alphabet[:axis],
            alphabet[axis:rank],
        )

        einsum_string_cols = '{}y{},z->{}{}zy'.format(
            alphabet[:axis],
            alphabet[axis:rank],
            alphabet[:axis],
            alphabet[axis:rank],
        )

        rows = np.einsum(
            einsum_string_rows,
            indices,
            ones,
        ).flatten()

        cols = np.einsum(
            einsum_string_cols,
            indices,
            ones,
        ).flatten()
        self.declare_partials(out_name, in1_name, rows=rows, cols=cols)

        rows = np.einsum(
            einsum_string_rows,
            indices,
            ones,
        ).flatten()

        cols = np.einsum(
            einsum_string_cols,
            indices,
            ones,
        ).flatten()
        self.declare_partials(out_name, in2_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        axis = self.options['axis']
        in1_name = self.options['in1_name']
        in2_name = self.options['in2_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.cross(
            inputs[in1_name],
            inputs[in2_name],
            axisa=axis,
            axisb=axis,
            axisc=axis,
        )

    def compute_partials(self, inputs, partials):
        shape = self.options['shape']
        axis = self.options['axis']
        in1_name = self.options['in1_name']
        in2_name = self.options['in2_name']
        out_name = self.options['out_name']

        ones = np.ones(3)
        eye = np.eye(3)
        rank = len(self.shape_without_axis)

        tmps = {0: None, 1: None, 2: None}
        for ind in range(3):
            array = np.einsum(
                '...,m->...m',
                np.ones(self.shape_without_axis),
                eye[ind, :],
            )

            array = np.einsum(
                '...,m->...m',
                np.cross(
                    np.einsum(
                        '...,m->...m',
                        np.ones(self.shape_without_axis),
                        eye[ind, :],
                    ),
                    inputs[in2_name],
                    axisa=-1,
                    axisb=axis,
                    axisc=-1,
                ),
                eye[ind, :],
            )

            tmps[ind] = array

        partials[out_name, in1_name] = (tmps[0] + tmps[1] + tmps[2]).flatten()

        tmps = {0: None, 1: None, 2: None}
        for ind in range(3):
            array = np.einsum(
                '...,m->...m',
                np.ones(self.shape_without_axis),
                eye[ind, :],
            )

            array = np.einsum(
                '...,m->...m',
                np.cross(
                    inputs[in1_name],
                    np.einsum(
                        '...,m->...m',
                        np.ones(self.shape_without_axis),
                        eye[ind, :],
                    ),
                    axisa=axis,
                    axisb=-1,
                    axisc=-1,
                ),
                eye[ind, :],
            )

            tmps[ind] = array

        partials[out_name, in2_name] = (tmps[0] + tmps[1] + tmps[2]).flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, 4, 3, 5)
    axis = 2

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('in1', val=np.random.random(shape))
    comp.add_output('in2', val=np.random.random(shape))
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = CrossProductComp(
        shape=shape,
        axis=axis,
        out_name='out',
        in1_name='in1',
        in2_name='in2',
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

import numpy as np

from openmdao.api import ExplicitComponent


class RotationMatrixComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('val', types=np.ndarray)
        self.options.declare('axis', types=str)
        self.options.declare('in_name',
                             types=str,
                             desc="This the name of the input angle tensor")
        self.options.declare('out_name',
                             types=str,
                             desc="The name of the rotation matrix tensor")

    def setup(self):
        shape = self.options['shape']
        val = self.options['val']
        if shape == (1, ):
            output_shape = (3, 3)

        else:
            output_shape = shape + (3, 3)

        axis = self.options['axis']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        self.add_input(in_name, shape=shape, val=val)
        self.add_output(out_name, shape=output_shape)

        rows = np.arange(np.prod(output_shape))
        cols = np.einsum('...,ij->...ij', np.arange(np.prod(shape)),
                         np.ones((3, 3), int)).flatten()
        self.declare_partials(out_name, in_name, rows=rows, cols=cols)

        if axis == 'x':
            self.i_cos1, self.j_cos1 = 1, 1
            self.i_cos2, self.j_cos2 = 2, 2
            self.i_sin1, self.j_sin1 = 1, 2
            self.i_sin2, self.j_sin2 = 2, 1
            self.i_one, self.j_one = 0, 0
        elif axis == 'y':
            self.i_cos1, self.j_cos1 = 0, 0
            self.i_cos2, self.j_cos2 = 2, 2
            self.i_sin1, self.j_sin1 = 2, 0
            self.i_sin2, self.j_sin2 = 0, 2
            self.i_one, self.j_one = 1, 1
        elif axis == 'z':
            self.i_cos1, self.j_cos1 = 0, 0
            self.i_cos2, self.j_cos2 = 1, 1
            self.i_sin1, self.j_sin1 = 0, 1
            self.i_sin2, self.j_sin2 = 1, 0
            self.i_one, self.j_one = 2, 2

    def compute(self, inputs, outputs):
        shape = self.options['shape']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        a = inputs[in_name]

        outputs[out_name] = 0
        outputs[out_name][..., self.i_cos1, self.j_cos1] = np.cos(a)
        outputs[out_name][..., self.i_cos2, self.j_cos2] = np.cos(a)
        outputs[out_name][..., self.i_sin1, self.j_sin1] = -np.sin(a)
        outputs[out_name][..., self.i_sin2, self.j_sin2] = np.sin(a)
        outputs[out_name][..., self.i_one, self.i_one] = 1.

    def compute_partials(self, inputs, partials):
        output_shape = self.options['shape'] + (3, 3)

        in_name = self.options['in_name']
        out_name = self.options['out_name']

        a = inputs[in_name]

        derivs = partials[out_name, in_name].reshape(output_shape)

        derivs[...] = 0.
        derivs[..., self.i_cos1, self.j_cos1] = -np.sin(a)
        derivs[..., self.i_cos2, self.j_cos2] = -np.sin(a)
        derivs[..., self.i_sin1, self.j_sin1] = -np.cos(a)
        derivs[..., self.i_sin2, self.j_sin2] = np.cos(a)


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, )

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('angle', val=np.random.rand(*shape))
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    for axis in ['x', 'y', 'z']:
        comp = RotationMatrixComp(
            shape=shape,
            axis=axis,
            in_name='angle',
            out_name='matrix_{}'.format(axis),
        )
        prob.model.add_subsystem('{}_comp'.format(axis), comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    prob.model.list_inputs(print_arrays=True)
    prob.model.list_outputs(print_arrays=True)

    if 0:
        for axis in ['x', 'y', 'z']:
            print(prob['matrix_{}'.format(axis)][0, 0, 0, :, :])

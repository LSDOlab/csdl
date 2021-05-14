import numpy as np
from openmdao.api import ExplicitComponent
from csdl.utils.process_options import name_types, get_names_list, shape_types, get_shapes_list, scalar_types, get_scalars_list


class LinearCombinationComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('out_name', types=str)
        self.options.declare('in_names',
                             default=None,
                             types=name_types,
                             allow_none=True)
        self.options.declare('shape', types=tuple)
        self.options.declare('coeffs', default=1., types=scalar_types)
        self.options.declare('coeffs_dict',
                             default=None,
                             types=dict,
                             allow_none=True)
        self.options.declare('constant',
                             default=0.,
                             types=(int, float, np.ndarray))
        self.options.declare('in_vals', types=list)

    def setup(self):

        shape = self.options['shape']

        if self.options['coeffs_dict']:
            self.options['in_names'] = []
            self.options['coeffs'] = []
            for in_name in self.options['coeffs_dict']:
                coeff = self.options['coeffs_dict'][in_name]
                self.options['in_names'].append(in_name)
                self.options['coeffs'].append(coeff)
        else:
            self.options['in_names'] = get_names_list(self.options['in_names'])
            self.options['coeffs'] = get_scalars_list(self.options['coeffs'],
                                                      self.options['in_names'])

        in_names = self.options['in_names']
        in_vals = self.options['in_vals']
        out_name = self.options['out_name']
        coeffs = self.options['coeffs']
        constant = self.options['constant']

        rows_cols = np.arange(np.prod(shape))

        self.add_output(out_name, shape=shape)
        for in_name, coeff, val in zip(in_names, coeffs, in_vals):
            self.add_input(in_name, shape=shape, val=val)
            self.declare_partials(out_name,
                                  in_name,
                                  val=coeff,
                                  rows=rows_cols,
                                  cols=rows_cols)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        coeffs = self.options['coeffs']
        constant = self.options['constant']

        outputs[out_name] = constant
        for in_name, coeff in zip(in_names, coeffs):
            outputs[out_name] += coeff * inputs[in_name]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, 3, 4)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape))
    comp.add_output('y', np.random.rand(*shape))
    comp.add_output('z', np.random.rand(*shape))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = LinearCombinationComp(
        shape=shape,
        in_names=['x', 'y', 'z'],
        out_name='f',
        coeffs=[1., -2., 3.],
        constant=1.5,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(1.5 + 1 * prob['x'] - 2 * prob['y'] + 3 * prob['z'] - prob['f'])

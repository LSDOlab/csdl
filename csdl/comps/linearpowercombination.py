import numpy as np
from openmdao.api import ExplicitComponent
from csdl.utils.process_options import name_types, get_names_list, shape_types, get_shapes_list, scalar_types, get_scalars_list


class LinearPowerCombinationComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('out_name', types=str)
        self.options.declare('in_names',
                             default=None,
                             types=name_types,
                             allow_none=True)
        self.options.declare('shape', types=tuple)
        self.options.declare('powers',
                             default=None,
                             types=np.ndarray,
                             allow_none=True)
        self.options.declare('terms_list',
                             default=None,
                             types=list,
                             allow_none=True)
        self.options.declare('constant',
                             default=0.,
                             types=(int, float, np.ndarray))
        self.options.declare('coeffs',
                             default=None,
                             types=(list, np.ndarray),
                             allow_none=True)

    def setup(self):

        if self.options['terms_list']:
            terms_list = self.options['terms_list']

            in_name_to_ivar = {}
            for coeff, power_dict in terms_list:
                in_name_to_ivar.update(power_dict)

            counter = 0
            for in_name in in_name_to_ivar:
                in_name_to_ivar[in_name] = counter
                counter += 1

            powers = np.zeros((len(terms_list), len(in_name_to_ivar)))
            for iterm, (coeff, power_dict) in enumerate(terms_list):
                for in_name in power_dict:
                    ivar = in_name_to_ivar[in_name]

                    powers[iterm, ivar] = power_dict[in_name]

            coeffs = np.zeros(len(terms_list))
            for iterm, (coeff, power_dict) in enumerate(terms_list):
                coeffs[iterm] = coeff

            in_names = []
            for in_name in in_name_to_ivar:
                in_names.append(in_name)

            self.options['in_names'] = in_names
            self.options['powers'] = powers
            self.options['coeffs'] = coeffs

        in_names = self.options['in_names']
        out_name = self.options['out_name']
        shape = self.options['shape']

        self.add_output(out_name, shape=shape)
        rows_cols = np.arange(np.prod(shape))

        for in_name in in_names:
            self.add_input(in_name, shape=shape)
            self.declare_partials(out_name,
                                  in_name,
                                  rows=rows_cols,
                                  cols=rows_cols)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        powers = self.options['powers']
        constant = self.options['constant']
        coeffs = self.options['coeffs']

        outputs[out_name] = constant
        for iterm in range(powers.shape[0]):
            term = coeffs[iterm] * np.ones(outputs[out_name].shape)
            for ivar, in_name in enumerate(in_names):
                power = powers[iterm, ivar]
                term *= inputs[in_name]**power

            outputs[out_name] += term

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        powers = self.options['powers']
        coeffs = self.options['coeffs']

        for in_name in in_names:
            deriv = np.zeros(self.options['shape'])

            for iterm in range(powers.shape[0]):
                term = coeffs[iterm]
                for ivar, in_name2 in enumerate(in_names):
                    power = powers[iterm, ivar]

                    a = 1.
                    b = power
                    if in_name == in_name2:
                        a = power
                        b = power - 1.

                    term *= a * inputs[in_name2]**b

                deriv += term

            partials[out_name, in_name] = deriv.flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, 3, 4)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape))
    comp.add_output('y', np.random.rand(*shape))
    comp.add_output('z', np.random.rand(*shape))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = LinearPowerCombinationComp(
        shape=shape,
        out_name='f',
        terms_list=[
            (1.5, dict(
                x=1.,
                y=2.,
                z=3.,
            )),
            (2., dict(
                x=4.5,
                y=5.,
                z=6.,
            )),
        ],
        constant=0.5,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    x = prob['x']
    y = prob['y']
    z = prob['z']
    print(0.5 + 1.5 * x**1 * y**2 * z**3 + 2. * x**4.5 * y**5 * z**6 -
          prob['f'])

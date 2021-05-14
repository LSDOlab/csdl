import numpy as np

from csdl.comps.array_explicit_component import ArrayExplicitComponent
from csdl.utils.process_options import name_types, get_names_list
from csdl.utils.process_options import scalar_types, get_scalars_list


class PowerCombinationComp(ArrayExplicitComponent):
    def array_initialize(self):
        self.options.declare('out_name', types=str)
        self.options.declare('in_names',
                             default=None,
                             types=name_types,
                             allow_none=True)
        self.options.declare('powers', default=1., types=scalar_types)
        self.options.declare('powers_dict',
                             default=None,
                             types=dict,
                             allow_none=True)
        self.options.declare('coeff',
                             default=1.,
                             types=(int, float, np.ndarray))
        self.options.declare('in_vals', types=list)

        self.post_initialize()

    def post_initialize(self):
        pass

    def pre_setup(self):
        pass

    def array_setup(self):
        self.pre_setup()

        if self.options['powers_dict']:
            self.options['in_names'] = []
            self.options['powers'] = []
            for in_name in self.options['powers_dict']:
                power = self.options['powers_dict'][in_name]
                self.options['in_names'].append(in_name)
                self.options['powers'].append(power)
        else:
            self.options['in_names'] = get_names_list(self.options['in_names'])
            self.options['powers'] = get_scalars_list(self.options['powers'],
                                                      self.options['in_names'])

        in_names = self.options['in_names']
        in_vals = self.options['in_vals']
        out_name = self.options['out_name']
        powers = self.options['powers']
        coeff = self.options['coeff']

        self.array_add_output(out_name)
        for in_name, val in zip(in_names, in_vals):
            self.array_add_input(in_name, val=val)
            self.array_declare_partials(out_name, in_name)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        powers = self.options['powers']
        coeff = self.options['coeff']

        outputs[out_name] = coeff
        for in_name, power in zip(in_names, powers):
            if np.any(inputs[in_name] == 0) and power < 0:
                print(in_name)
                print(inputs[in_name])
                exit()
            outputs[out_name] *= inputs[in_name]**power

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        powers = self.options['powers']
        coeff = self.options['coeff']

        value = coeff
        for in_name, power in zip(in_names, powers):
            value *= inputs[in_name]**power

        for in_name in in_names:
            deriv = coeff * np.ones(self.options['shape'])
            for in_name2, power in zip(in_names, powers):
                a = 1.
                b = power
                if in_name == in_name2:
                    a = power
                    b = power - 1.
                deriv *= a * inputs[in_name2]**b

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

    # comp = PowerCombinationComp(
    #     shape=shape,
    #     out_name='f',
    #     in_names=['x', 'y', 'z'],
    #     powers=[1., -2., 3.],
    #     coeff=1.5,
    # )
    # prob.model.add_subsystem('comp', comp, promotes=['*'])

    comp = PowerCombinationComp(
        shape=shape,
        out_name='f',
        # coeff=1.5,
        powers_dict=dict(x=0.5,
                         # y=-2.,
                         # z=3.,
                         ))
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(1.5 * prob['x']**1 * prob['y']**-2 * prob['z']**3 - prob['f'])

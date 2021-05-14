import numpy as np
from copy import deepcopy
import time
from guppy import hpy

from openmdao.api import ExplicitComponent
from csdl.utils.process_options import name_types, get_names_list, shape_types, get_shapes_list
from csdl.utils.einsum_utils import compute_einsum_shape


class EinsumComp(ExplicitComponent):
    """
    This component computes the Einstein summation convention applied on input components that are arrays.
    Partial derivatives computed are always stored in dense format.

    Options
    -------
    in_names: name_types (str or list of str)
        Input component names that represent input arrays.
        Can be a string or a list with a single string when there is only one input array.
    in_shapes: shape_types (tuple or list of tuples)
        Input component shapes that, in order, represents the shapes of the input arrays.
        Can be a tuple or a list with a single tuple when there is only one input array.
    operation: str
        Einstein summation operation written using subscript labels seperated by commas. The last subscript label should represent the output form and it should be seperated from the input subscript labels using the indicator '->'. All subscripts of tensors should be named using an alphabet (numpy convention of '...' to represent multiple subscipts is not permitted).
    out_name: str
        Output component name that represents the output array calculated based on the Einstein summation operation given.
    """
    def initialize(self):
        self.options.declare('in_names',
                             default=None,
                             types=name_types,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shapes', types=shape_types)
        # Nametypes might be a string or a list
        self.options.declare('operation', types=str)
        self.options.declare('out_shape', types=tuple, default=None)
        self.options.declare('in_vals', types=list)

    # Add inputs and output, and declare partials
    def setup(self):
        operation = self.options['operation']

        # Changes from a string to a list with one element if there was only one input
        self.options['in_names'] = get_names_list(self.options['in_names'])
        self.options['in_shapes'] = get_shapes_list(self.options['in_shapes'])

        in_names = self.options['in_names']
        in_shapes = self.options['in_shapes']
        out_name = self.options['out_name']
        out_shape = self.options['out_shape']
        in_vals = self.options['in_vals']

        # Find unused characters in operation
        check_string = 'abcdefghijklmnopqrstuvwxyz'
        self.unused_chars = ''
        for char in check_string:
            if not (char in operation):
                self.unused_chars += char

        # Translate the operation string into a list
        self.operation_aslist = []

        # Representation of each tensor in the operation string
        tensor_rep = ''
        for char in operation:
            if char.isalpha():
                tensor_rep += char
            elif (char == ',' or char == '-'):
                self.operation_aslist.append(tensor_rep)
                tensor_rep = ''

        # When output is a scalar
        if operation[-1] == '>':
            self.operation_aslist.append(tensor_rep)

        # When output is a tensor
        else:
            self.operation_aslist.append(tensor_rep)

        # When output shape is not provided
        if out_shape == None:
            self.out_shape = compute_einsum_shape(
                self.operation_aslist,
                in_shapes,
            )
        else:
            self.out_shape = out_shape

        if self.out_shape == (1, ):
            self.add_output(out_name)
        else:
            self.add_output(out_name, shape=self.out_shape)

        completed_in_names = []
        self.I = []
        operation_aslist = self.operation_aslist

        for in_name_index, (in_name,
                            in_val) in enumerate(zip(in_names, in_vals)):
            if in_name in completed_in_names:
                continue
            else:
                completed_in_names.append(in_name)
            self.add_input(in_name, shape=in_shapes[in_name_index], val=in_val)
            self.declare_partials(out_name, in_name)

            shape = in_shapes[in_name_index]
            size = np.prod(shape)
            rank = len(shape)
            flat_indices = np.arange(size)
            ind = np.unravel_index(flat_indices, shape)
            # self.list_of_tuple_of_indices_of_input_tensors.append(ind)

            # Generate I efficiently for each in_name

            I_shape = 2 * list(shape)
            I_shape = tuple(I_shape)
            I_ind = 2 * list(ind)
            I_ind = tuple(I_ind)

            I = np.zeros(I_shape)
            I[I_ind] += 1

            self.I.append(I)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        operation = self.options['operation']

        outputs[out_name] = np.einsum(
            operation, *(inputs[in_name] for in_name in in_names))

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        in_shapes = self.options['in_shapes']
        out_name = self.options['out_name']
        operation = self.options['operation']

        unused_chars = self.unused_chars
        operation_aslist = self.operation_aslist

        completed_in_names = []

        for in_name_index, in_name in enumerate(in_names):
            '''Checking if we are at a repeated input whose derivative was computed at its first occurence in the in_names. If true, we will skip the current iteration of in_name'''
            if in_name in completed_in_names:
                continue
            else:
                completed_in_names.append(in_name)

            shape = in_shapes[in_name_index]
            size = np.prod(shape)
            rank = len(shape)

            # Compute the locations where the same input is used
            locations = []
            for idx, same_name in enumerate(in_names):
                if same_name == in_name:
                    locations.append(idx)

            new_in_name_tensor_rep = operation_aslist[in_name_index]
            new_in_name_tensor_rep += unused_chars[:rank]
            new_output_tensor_rep = operation_aslist[-1]
            new_output_tensor_rep += unused_chars[:rank]

            new_operation_aslist = deepcopy(operation_aslist)
            new_operation_aslist[in_name_index] = new_in_name_tensor_rep
            new_operation_aslist[-1] = new_output_tensor_rep

            # Compute new_operation by replacing each tensor_rep for in_name in first location by I's tensor_rep
            new_operation = ''
            for string_rep in new_operation_aslist[:-1]:
                new_operation += string_rep
                new_operation += ','
            new_operation = new_operation[:-1] + '->'
            new_operation += new_operation_aslist[-1]

            partials[out_name, in_name] = np.einsum(
                new_operation,
                *(inputs[in_name] for in_name in in_names[:in_name_index]),
                self.I[in_name_index],
                *(inputs[in_name] for in_name in in_names[in_name_index + 1:]))

            for i in locations[1:]:
                new_operation_aslist = deepcopy(operation_aslist)
                new_operation_aslist[
                    i] = operation_aslist[i] + unused_chars[:rank]
                new_operation_aslist[-1] = new_output_tensor_rep

                new_operation = ''
                for string_rep in new_operation_aslist[:-1]:
                    new_operation += string_rep
                    new_operation += ','
                new_operation = new_operation[:-1] + '->'
                new_operation += new_operation_aslist[-1]

                partials[out_name, in_name] += np.einsum(
                    new_operation,
                    *(inputs[in_name] for in_name in in_names[:i]),
                    self.I[len(completed_in_names) - 1],
                    *(inputs[in_name]
                      for in_name in in_names[i + 1:])).reshape(
                          partials[out_name, in_name].shape)


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    # h = hpy()

    shape1 = (2, 2, 4)
    shape2 = (2, 7, 4)
    shape3 = (7, 2, 4)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape1))
    comp.add_output('y', np.random.rand(*shape2))
    comp.add_output('z', np.random.rand(*shape3))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])
    # out_shape = (2, 3, 4, 4, 7)

    comp = EinsumComp(
        in_names=['x', 'y', 'x'],
        in_shapes=[(2, 2, 4), (2, 7, 4), (2, 2, 4)],
        out_name='f',
        operation='abc,ade,fae->bcdfa',
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    start = time.time()
    # h.setrelheap()

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    # mem = h.heap()
    end = time.time()

    # print(end - start)
    # print(mem.size / 1024 / 1024)

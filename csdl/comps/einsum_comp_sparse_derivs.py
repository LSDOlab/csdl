import numpy as np
from copy import deepcopy
import time
from guppy import hpy

from openmdao.api import ExplicitComponent
from csdl.utils.process_options import name_types, get_names_list, shape_types, get_shapes_list
from csdl.utils.einsum_utils import compute_einsum_shape


class SparsePartialEinsumComp(ExplicitComponent):
    """
    This component computes the Einstein summation convention applied on input components that are arrays.
    In addition, the sparsity structure of the partial derivatives of the output with respect to each input is precomputed in the setup and partials are always stored in sparse format.

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
        for idx, (in_name, in_val) in enumerate(zip(in_names, in_vals)):
            if in_name in completed_in_names:
                continue
            else:
                completed_in_names.append(in_name)
            self.add_input(in_name, shape=in_shapes[idx], val=in_val)
            self.declare_partials(out_name, in_name)

        # List contains tuples, each tuple contains all the indices for a given input tensor
        self.list_of_tuple_of_indices_of_input_tensors = []
        self.surviving_axes_map = []
        completed_in_names = []
        self.I = []
        self.partial_indices_list = []
        self.flattened_indices_list = []
        operation_aslist = self.operation_aslist
        self.sparsity_partials = [False] * len(in_names)

        for in_name_index, in_name in enumerate(in_names):

            # Map locations of axes from the input to that in the output for axes that does not get nullified in the output (for computing indices of nonzeros in the partials)
            # Note: without any surviving axes, the partial with respect to that input would be dense.
            surviving_axes_map = {}
            output_tensor_rep = operation_aslist[-1]
            for idx, axis in enumerate(operation_aslist[in_name_index]):
                if axis in output_tensor_rep:
                    surviving_axes_map[idx] = output_tensor_rep.index(axis)

            self.surviving_axes_map.append(surviving_axes_map)

        completed_in_names = []
        for in_name_index, in_name in enumerate(in_names):

            if in_name in completed_in_names:
                continue
            else:
                completed_in_names.append(in_name)

            shape = in_shapes[in_name_index]
            size = np.prod(shape)
            rank = len(shape)
            flat_indices = np.arange(size)
            ind = np.unravel_index(flat_indices, shape)
            self.list_of_tuple_of_indices_of_input_tensors.append(ind)

            # Generate I efficiently for each in_name

            I_shape = 2 * list(shape)
            I_shape = tuple(I_shape)
            I_ind = 2 * list(ind)
            I_ind = tuple(I_ind)

            I = np.zeros(I_shape)
            I[I_ind] += 1

            self.I.append(I)

            locations = []
            for idx, same_name in enumerate(in_names):
                if same_name == in_name:
                    locations.append(idx)
            '''
            When tried to compute indices in just one go

            # surviving_axes_maps_for_in_name = [self.surviving_axes_map[k] for k in locations]

            # common_surviving_axes = deepcopy(surviving_axes_maps_for_in_name[0])
            # for axis in common_surviving_axes:
                # common_surviving_axes[axis] = [common_surviving_axes[axis]]

            # for idx in range(len(operation_aslist[in_name_index])):
            #     for index, loc in enumerate(locations[1:]):
            #         if idx not in surviving_axes_maps_for_in_name[index+1]:
            #             common_surviving_axes.pop(idx, None) # Delete axes that are not surviving in all reps of a given in_name
            #             break
            #         elif surviving_axes_maps_for_in_name[index+1][idx] not in common_surviving_axes[idx]:
            #             common_surviving_axes[idx].append(surviving_axes_maps_for_in_name[index+1][idx])

            '''

            flattened_input_index = np.array([
                0,
            ])
            flattened_output_index = np.array([
                0,
            ])

            out_shape = self.out_shape
            new_output_tensor_rep = operation_aslist[
                -1] + self.unused_chars[:len(operation_aslist[in_name_index])]
            partial_rank = len(new_output_tensor_rep)

            full_partial_indices_list = [np.array([
                0,
            ])] * partial_rank

            for loc in locations:
                if self.surviving_axes_map[loc]:
                    self.sparsity_partials[loc] = True

            sparse_partial = True
            for loc in locations:
                if self.sparsity_partials[loc] == False:
                    sparse_partial = False
                    break

            if sparse_partial:
                for loc in locations:
                    # Partials are dense if surviving_axes_map is empty
                    if not (self.surviving_axes_map[loc]):
                        sparse_partial
                    if self.surviving_axes_map[loc]:
                        remainder_output_shape = []
                        partial_indices_list = np.arange(partial_rank).tolist()
                        partial_indices_list[-rank:] = ind
                        tensor_rep = operation_aslist[loc]

                        # surviving_axes_index_in_output = [item for sublist in common_surviving_axes.values() for item in sublist]

                        for idx, axis in enumerate(output_tensor_rep):
                            if not (idx
                                    in self.surviving_axes_map[loc].values()):
                                remainder_output_shape.append(out_shape[idx])
                            else:
                                partial_indices_list[output_tensor_rep.index(
                                    axis)] = ind[operation_aslist[loc].index(
                                        axis)]
                            '''
                            When tried to compute indices in just one go

                            # else:
                            #     for loc in locations:
                            #         if axis in operation_aslist[loc]:
                            #             partial_indices_list[output_tensor_rep.index(axis)] = ind[operation_aslist[loc].index(axis)]
                            #             break
                            '''

                        remainder_output_size = 1
                        # Computing remainder indices if remainder_output_shape is not empty
                        if remainder_output_shape:
                            remainder_output_shape = tuple(
                                remainder_output_shape)
                            remainder_output_size = np.prod(
                                remainder_output_shape)
                            remainder_flat_indices = np.arange(
                                remainder_output_size)
                            remainder_output_ind = np.unravel_index(
                                remainder_flat_indices, remainder_output_shape)

                        i = 0
                        for idx, indices in enumerate(partial_indices_list):
                            if isinstance(indices, np.ndarray):
                                partial_indices_list[idx] = np.tile(
                                    indices, remainder_output_size)
                            elif isinstance(indices, int):
                                partial_indices_list[idx] = np.repeat(
                                    remainder_output_ind[i], size)
                                i += 1
                            else:
                                raise Exception()

                        for k in range(partial_rank):
                            full_partial_indices_list[k] = np.append(
                                full_partial_indices_list[k],
                                partial_indices_list[k])

                partial_size = full_partial_indices_list[0].size
                full_partial_indices_array = full_partial_indices_list[
                    0].reshape((partial_size, 1))

                for k in range(1, partial_rank):
                    full_partial_indices_array = np.append(
                        full_partial_indices_array,
                        full_partial_indices_list[k].reshape(
                            (partial_size, 1)),
                        axis=1)
                full_partial_indices_array = np.unique(
                    full_partial_indices_array, axis=0)
                full_partial_indices_list = [
                    full_partial_indices_array[:, x]
                    for x in range(partial_rank)
                ]

                flattened_input_index = np.ravel_multi_index(
                    full_partial_indices_list[-rank:], shape)
                flattened_output_index = np.ravel_multi_index(
                    full_partial_indices_list[:-rank], out_shape)

                self.declare_partials(out_name,
                                      in_name,
                                      rows=flattened_output_index,
                                      cols=flattened_input_index)
                self.partial_indices_list.append(full_partial_indices_list)
                self.flattened_indices_list.append(
                    (flattened_input_index, flattened_output_index))

            else:
                self.declare_partials(out_name, in_name)

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

            # Calculate new_operation for each in_name in first location

            # Compute the locations where the same input is used
            locations = []
            for idx, same_name in enumerate(in_names):
                if same_name == in_name:
                    locations.append(idx)

            in_name_tensor_rep = operation_aslist[in_name_index]
            rank = len(in_name_tensor_rep)
            output_tensor_rep = operation_aslist[-1]

            new_in_name_tensor_rep = in_name_tensor_rep + unused_chars[:rank]
            new_output_tensor_rep = output_tensor_rep + unused_chars[:rank]

            new_operation_aslist = deepcopy(operation_aslist)
            new_operation_aslist[in_name_index] = new_in_name_tensor_rep
            new_operation_aslist[-1] = new_output_tensor_rep

            # Compute new_in_names by replacing each in_name in first location by I
            new_operation = ''
            for string_rep in new_operation_aslist[:-1]:
                new_operation += string_rep
                new_operation += ','
            new_operation = new_operation[:-1] + '->'
            new_operation += new_operation_aslist[-1]

            partial = np.einsum(
                new_operation,
                *(inputs[in_name] for in_name in in_names[:in_name_index]),
                self.I[in_name_index],
                *(inputs[in_name] for in_name in in_names[in_name_index + 1:]))

            for i in locations[1:]:
                new_operation_aslist = deepcopy(operation_aslist)
                new_in_name_tensor_rep = operation_aslist[
                    i] + unused_chars[:rank]
                new_operation_aslist[i] = new_in_name_tensor_rep
                new_operation_aslist[-1] = new_output_tensor_rep

                new_operation = ''
                for string_rep in new_operation_aslist[:-1]:
                    new_operation += string_rep
                    new_operation += ','
                new_operation = new_operation[:-1] + '->'
                new_operation += new_operation_aslist[-1]

                partial += np.einsum(
                    new_operation,
                    *(inputs[in_name] for in_name in in_names[:i]),
                    self.I[len(completed_in_names) - 1],
                    *(inputs[in_name] for in_name in in_names[i + 1:]))

            sparse_partial = True
            for i in locations:
                if self.sparsity_partials[i] == False:
                    sparse_partial = False

            # Partials are dense if surviving_axes_map is empty for at least one location
            if sparse_partial:
                partial_indices_list = self.partial_indices_list[
                    len(completed_in_names) - 1]

                partials[out_name, in_name] = partial[partial_indices_list]

            else:
                partials[out_name, in_name] = partial


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    # h = hpy()

    shape1 = (2, 2, 4)
    shape2 = (2, 7, 4)
    shape3 = (2, 2, 4)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape1))
    comp.add_output('y', np.random.rand(*shape2))
    comp.add_output('z', np.random.rand(*shape3))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = SparsePartialEinsumComp(
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

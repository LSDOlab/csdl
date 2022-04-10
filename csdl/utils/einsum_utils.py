from typing import List, Tuple


def compute_einsum_shape(
    operation_aslist: List[str],
    in_shapes: Tuple[int] | List[Tuple[int]],
):
    out_shape = []

    if operation_aslist[-1] == '':
        out_shape = (1, )

    else:
        for char in operation_aslist[-1]:
            for i, tensor_rep in enumerate(operation_aslist[:-1]):
                if (char in tensor_rep):
                    shape_ind = tensor_rep.index(char)
                    out_shape.append(in_shapes[i][shape_ind])
                    break

        out_shape = tuple(out_shape)

    return out_shape


def einsum_subscripts_tolist(subscripts: str):
    operation_aslist = []
    tensor_rep = ''
    for char in subscripts:
        if char.isalpha():
            tensor_rep += char
        elif (char == ',' or char == '-'):
            operation_aslist.append(tensor_rep)
            tensor_rep = ''

    # When output is a scalar
    if subscripts[-1] == '>':
        operation_aslist.append(tensor_rep)

    # When output is a tensor
    else:
        operation_aslist.append(tensor_rep)

    return operation_aslist


def new_einsum_subscripts_to_string_and_list(
    subscripts: List[Tuple],
    scalar_output=False,
):
    # Assign characters to each axis_name in the tuples
    unused_chars = 'abcdefghijklmnopqrstuvwxyz'
    axis_map = {}
    operation_as_string = ''
    operation_aslist = []

    if not (scalar_output):
        num_inputs = len(subscripts) - 1
    else:
        num_inputs = len(subscripts)

    for axis_names in subscripts[:num_inputs]:
        tensor_rep = ''

        # Mapping an alphabet for each axis in the tuple
        for axis in axis_names:
            if not (axis in axis_map):
                axis_map[axis] = unused_chars[0]
                unused_chars = unused_chars[1:]
            tensor_rep += axis_map[axis]

        operation_as_string += tensor_rep
        operation_as_string += ','
        operation_aslist.append(tensor_rep)

    tensor_rep = ''

    # When output is a tensor
    if len(subscripts) == (num_inputs + 1):
        for axis in subscripts[-1]:
            tensor_rep += axis_map[axis]

    operation_as_string = operation_as_string[:-1] + '->'
    operation_as_string += tensor_rep
    operation_aslist.append(tensor_rep)

    return operation_aslist, operation_as_string

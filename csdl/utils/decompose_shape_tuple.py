def decompose_shape_tuple(shape, select_indices):
    alphabet = 'abcdefghij'

    einsum_selection = ''
    einsum_full = ''
    einsum_remainder = ''
    shape_selection = []
    shape_full = []
    shape_remainder = []
    for index in range(len(shape)):
        if index not in select_indices:
            einsum_selection += alphabet[index]
            shape_selection.append(shape[index])
        else:
            einsum_remainder += alphabet[index]
            shape_remainder.append(shape[index])
        einsum_full += alphabet[index]
        shape_full.append(shape[index])

    shape_selection = tuple(shape_selection)
    shape_full = tuple(shape_full)
    shape_remainder = tuple(shape_remainder)

    return (
        einsum_selection, einsum_remainder, einsum_full,
        shape_selection, shape_remainder, shape_full,
    )
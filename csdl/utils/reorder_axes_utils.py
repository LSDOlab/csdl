
def compute_new_axes_locations(in_shape: tuple, operation: str):
        operation = operation.replace(" ", "")
        input_subscript = operation.split('->')[0]
        output_subscript = operation.split('->')[1]

        new_axes_locations = []
        for i in range(len(output_subscript)):
            new_axes_locations.append(input_subscript.index(output_subscript[i]))

        return new_axes_locations 
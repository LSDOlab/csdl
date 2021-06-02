import numpy as np


def structure_data(inputs, output, ny=1):
    """
    Structure data for training a surrogate model
    output must be the same size as all inputs combined
    """
    # FIXME: get correct xlimits
    # xlimits = np.zeros((len(inputs), 2))
    # for i in range(len(inputs)):
    # xlimits[i, 0] = np.amin(inputs[i])
    # xlimits[i, 1] = np.amax(inputs[i])

    print((int(np.prod(output.shape) / ny), len(inputs)))

    xt = np.array(np.meshgrid(
        *inputs,
        indexing='ij',
    ), ).reshape(
        int(np.prod(output.shape) / ny),
        len(inputs),
    )

    # KLUDGE: get xlimits based on structured training inputs
    xlimits = np.zeros((len(inputs), 2))
    xlimits[:, 0] = np.min(xt) * np.ones(len(inputs))
    xlimits[:, 1] = np.max(xt) * np.ones(len(inputs))

    return xt, xlimits

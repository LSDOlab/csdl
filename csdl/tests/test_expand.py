import numpy as np
import pytest


def test_expand_scalar2array(backend):
    from csdl.examples.valid.ex_expand_scalar2_array import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_array_equal(sim['scalar'], np.array([1]))
    np.testing.assert_array_equal(
        sim['expanded_scalar'],
        np.array([
            [1., 1., 1.],
            [1., 1., 1.],
        ]),
    )

    # result = sim.check_partials(out_stream=None,
    # compact_print=True,
    # method='cs')
    # sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_expand_array2higherarray(backend):
    from csdl.examples.valid.ex_expand_array2_higher_array import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    array = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
    ])
    expanded_array = np.empty((2, 4, 3, 1))
    for i in range(4):
        for j in range(1):
            expanded_array[:, i, :, j] = array

    np.testing.assert_array_equal(sim['array'], array)
    np.testing.assert_array_equal(sim['expanded_array'], expanded_array)

    # result = sim.check_partials(out_stream=None,
    # compact_print=True,
    # method='cs')
    # sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_scalar_incorrect_order(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_expand_scalar_incorrect_order import example
    with pytest.raises(TypeError):
        example(eval('Simulator'))


def test_no_indices(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_expand_array_no_indices import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_array_invalid_indices1(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_expand_array_invalid_indices1 import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_array_invalid_indices2(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_expand_array_invalid_indices2 import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))

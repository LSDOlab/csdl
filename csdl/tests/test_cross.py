import numpy as np
import pytest


def test_vector_vector_cross(backend):
    from csdl.examples.valid.ex_cross_vector_vector import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    vec1 = np.arange(3)
    vec2 = np.arange(3) + 1

    desired_output = np.cross(vec1, vec2)
    np.testing.assert_almost_equal(sim['VecVecCross'], desired_output)

    partials_error = sim.check_partials(includes=['comp_VecVecCross'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_cross(backend):
    from csdl.examples.valid.ex_cross_tensor_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    shape = (2, 5, 4, 3)
    num_elements = np.prod(shape)

    ten1 = np.arange(num_elements).reshape(shape)
    ten2 = np.arange(num_elements).reshape(shape) + 6

    desired_output = np.cross(ten1, ten2, axis=3)
    np.testing.assert_almost_equal(sim['TenTenCross'], desired_output)

    partials_error = sim.check_partials(includes=['comp_TenTenCross'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_cross_different_shapes(backend):
    from csdl.examples.invalid.ex_cross_different_shapes import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(ValueError):
        sim = example(eval('Simulator'))


def test_cross_incorrect_axis_index(backend):
    from csdl.examples.invalid.ex_cross_incorrect_axis_index import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(ValueError):
        sim = example(eval('Simulator'))

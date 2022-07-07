import numpy as np
import pytest


def test_vector_vector_outer(backend):
    from csdl.examples.valid.ex_outer_vector_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    m = 3

    # Values for the two vectors
    vec1 = np.arange(m)
    vec2 = np.arange(m, 2 * m)

    # VEC VEC OUTER
    desired_output = np.outer(vec1, vec2)
    np.testing.assert_almost_equal(sim['VecVecOuter'], desired_output)

    partials_error = sim.check_partials(includes=['comp_VecVecOuter'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tensor_vector_outer(backend):
    from csdl.examples.valid.ex_outer_tensor_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    m = 3
    n = 4
    p = 5

    # Shape of the vectors
    vec_shape = (m, )

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Values for the two vectors
    vec1 = np.arange(m)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)

    # TENSOR VECTOR OUTER
    desired_output = np.einsum('ijk,l->ijkl', ten1, vec1)
    np.testing.assert_almost_equal(sim['TenVecOuter'], desired_output)

    partials_error = sim.check_partials(includes=['comp_TenVecOuter'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tensor_tensor_outer(backend):
    from csdl.examples.valid.ex_outer_tensor_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    m = 3
    n = 4
    p = 5

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # TENSOR TENSOR OUTER
    desired_output = np.einsum('ijk,lmn->ijklmn', ten1, ten2)
    np.testing.assert_almost_equal(sim['TenTenOuter'], desired_output)

    partials_error = sim.check_partials(includes=['comp_TenTenOuter'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)

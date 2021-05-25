from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_matrix_matrix_multiplication_matmat(name):
    from csdl.examples.valid.ex_matmat_mat_mat_product import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))

    m = 3
    n = 2
    p = 4

    # Shape of the first matrix (3,2)
    shape1 = (m, n)

    # Shape of the second matrix (2,4)
    shape2 = (n, p)

    # Creating the values of both matrices
    mat1 = np.arange(m * n).reshape(shape1)
    mat2 = np.arange(n * p).reshape(shape2)

    # Creating the values for the vector
    vec1 = np.arange(n)

    # MATRIX MATRIX MULTIPLICATION
    desired_output = np.matmul(mat1, mat2)
    np.testing.assert_almost_equal(sim['MatMat'], desired_output)

    partials_error = sim.check_partials(includes=['comp_MatMat'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_matrix_vector_multiplication_matmat(name):
    from csdl.examples.valid.ex_matmat_mat_vec_product import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))

    m = 3
    n = 2
    p = 4

    # Shape of the first matrix (3,2)
    shape1 = (m, n)

    # Shape of the second matrix (2,4)
    shape2 = (n, )

    # Creating the values of both matrices
    mat1 = np.arange(m * n).reshape(shape1)
    vec1 = np.arange(n).reshape(shape2)

    # MATRIX MATRIX MULTIPLICATION
    desired_output = np.matmul(mat1, vec1)
    np.testing.assert_almost_equal(sim['MatVec'], desired_output)

    partials_error = sim.check_partials(includes=['comp_MatVec'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_matrix_matrix_incompatible_shapes(name):
    with pytest.raises(Exception):
        from csdl.examples.invalid.ex_matmat_matrix_matrix_incompatible_shapes import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))

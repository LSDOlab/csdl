import numpy as np
from numpy.testing._private.utils import assert_
import pytest


def test_sum_single_vector(backend):
    from csdl.examples.valid.ex_sum_single_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3

    v1 = np.arange(n)

    desired_vector_sum = np.sum(v1)

    np.testing.assert_almost_equal(sim['single_vector_sum'],
                                   desired_vector_sum)
    assert sim['v1'].shape == (n, )
    assert sim['single_vector_sum'].shape == (1, )

    partials_error_vector_sum = sim.check_partials(
        includes=['comp_single_vector_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_vector_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_single_matrix(backend):
    from csdl.examples.valid.ex_sum_single_matrix import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_matrix_sum = np.sum(M1)

    np.testing.assert_almost_equal(sim['single_matrix_sum'],
                                   desired_matrix_sum)
    assert sim['M1'].shape == (n, m)
    assert sim['single_matrix_sum'].shape == (1, )

    partials_error_vector_sum = sim.check_partials(
        includes=['comp_single_matrix_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_vector_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_single_tensor(backend):
    from csdl.examples.valid.ex_sum_single_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 4
    p = 5
    q = 6

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))

    desired_tensor_sum = np.sum(T1)

    np.testing.assert_almost_equal(sim['single_tensor_sum'],
                                   desired_tensor_sum)
    assert sim['single_tensor_sum'].shape == (1, )

    partials_error_tensor_sum = sim.check_partials(
        includes=['comp_single_tensor_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_tensor_sum,
                              atol=1.e-5,
                              rtol=1.e-5)


def test_sum_multiple_vector(backend):
    from csdl.examples.valid.ex_sum_multiple_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3

    v1 = np.arange(n)
    v2 = np.arange(n, 2 * n)

    desired_vector_sum = v1 + v2

    np.testing.assert_almost_equal(sim['multiple_vector_sum'],
                                   desired_vector_sum)
    assert sim['multiple_vector_sum'].shape == (n, )

    partials_error_vector_sum = sim.check_partials(
        includes=['comp_multiple_vector_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_vector_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_matrix(backend):
    from csdl.examples.valid.ex_sum_multiple_matrix import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_matrix_sum = M1 + M2

    np.testing.assert_almost_equal(sim['multiple_matrix_sum'],
                                   desired_matrix_sum)
    assert sim['multiple_matrix_sum'].shape == (n, m)

    partials_error_matrix_sum = sim.check_partials(
        includes=['comp_multiple_matrix_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_matrix_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_tensor(backend):
    from csdl.examples.valid.ex_sum_multiple_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    p = 7
    q = 10

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))
    T2 = np.arange(n * m * p * q, 2 * n * m * p * q).reshape(
        (n, m, p, q))

    desired_tensor_sum = T1 + T2

    np.testing.assert_almost_equal(sim['multiple_tensor_sum'],
                                   desired_tensor_sum)
    assert sim['multiple_tensor_sum'].shape == (n, m, p, q)

    partials_error_tensor_sum = sim.check_partials(
        includes=['comp_multiple_tensor_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_tensor_sum,
                              atol=1.e-5,
                              rtol=1.e-5)


def test_sum_single_matrix_along0(backend):
    from csdl.examples.valid.ex_sum_single_matrix_along0 import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_single_matrix_sum_axis_0 = np.sum(M1, axis=0)

    np.testing.assert_almost_equal(sim['single_matrix_sum_along_0'],
                                   desired_single_matrix_sum_axis_0)
    assert sim['single_matrix_sum_along_0'].shape == (m, )

    partials_error_single_matrix_axis_0 = sim.check_partials(
        includes=['comp_single_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_single_matrix_axis_0,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_single_matrix_along1(backend):
    from csdl.examples.valid.ex_sum_single_matrix_along1 import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_single_matrix_sum_axis_1 = np.sum(M1, axis=1)

    np.testing.assert_almost_equal(sim['single_matrix_sum_along_1'],
                                   desired_single_matrix_sum_axis_1)
    assert sim['single_matrix_sum_along_1'].shape == (n, )

    partials_error_single_matrix_axis_1 = sim.check_partials(
        includes=['comp_single_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_single_matrix_axis_1,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_matrix_along0(backend):
    from csdl.examples.valid.ex_sum_multiple_matrix_along0 import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_multiple_matrix_sum_axis_0 = np.sum(M1 + M2, axis=0)

    np.testing.assert_almost_equal(sim['multiple_matrix_sum_along_0'],
                                   desired_multiple_matrix_sum_axis_0)

    partials_error_multiple_matrix_axis_0 = sim.check_partials(
        includes=['comp_multiple_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_0,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_matrix_along1(backend):
    from csdl.examples.valid.ex_sum_multiple_matrix_along1 import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_multiple_matrix_sum_axis_1 = np.sum(M1 + M2, axis=1)

    np.testing.assert_almost_equal(sim['multiple_matrix_sum_along_1'],
                                   desired_multiple_matrix_sum_axis_1)

    partials_error_multiple_matrix_axis_1 = sim.check_partials(
        includes=['comp_multiple_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_1,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_concatenate_sums(backend):
    from csdl.examples.valid.ex_sum_concatenate import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    x = np.array([np.sum(np.arange(5)), np.sum(np.arange(4)), 0])

    np.testing.assert_almost_equal(sim['sum_vector'], x)
    np.testing.assert_almost_equal(
        sim['single_vector_sum_1a'],
        sim['single_vector_sum_1a'],
    )

    partials_error_multiple_matrix_axis_1 = sim.check_partials(
        out_stream=None, compact_print=True, method='fd')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_1,
                              atol=1.e-8,
                              rtol=1.e-8)
    partials_error_multiple_matrix_axis_1 = sim.check_partials(
        out_stream=None, compact_print=True, method='cs')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_1,
                              atol=1.e-8,
                              rtol=1.e-8)


def test_sum_single_vector_random(backend):
    from csdl.examples.valid.ex_sum_single_vector_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    np.random.seed(0)

    v1 = np.random.rand(n)

    desired_vector_sum = np.sum(v1)

    np.testing.assert_almost_equal(sim['single_vector_sum'],
                                   desired_vector_sum)
    assert sim['v1'].shape == (n, )
    assert sim['single_vector_sum'].shape == (1, )

    partials_error_vector_sum = sim.check_partials(
        includes=['comp_single_vector_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_vector_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_single_matrix_random(backend):
    from csdl.examples.valid.ex_sum_single_matrix_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    np.random.seed(0)

    M1 = np.random.rand(n * m).reshape((n, m))

    desired_matrix_sum = np.sum(M1)

    np.testing.assert_almost_equal(sim['single_matrix_sum'],
                                   desired_matrix_sum)
    assert sim['M1'].shape == (n, m)
    assert sim['single_matrix_sum'].shape == (1, )

    partials_error_vector_sum = sim.check_partials(
        includes=['comp_single_matrix_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_vector_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_single_tensor_random(backend):
    from csdl.examples.valid.ex_sum_single_tensor_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 4
    p = 5
    q = 6
    np.random.seed(0)

    T1 = np.random.rand(n * m * p * q).reshape((n, m, p, q))

    desired_tensor_sum = np.sum(T1)

    np.testing.assert_almost_equal(sim['single_tensor_sum'],
                                   desired_tensor_sum)
    assert sim['single_tensor_sum'].shape == (1, )

    partials_error_tensor_sum = sim.check_partials(
        includes=['comp_single_tensor_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_tensor_sum,
                              atol=1.e-5,
                              rtol=1.e-5)


def test_sum_multiple_vector_random(backend):
    from csdl.examples.valid.ex_sum_multiple_vector_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    np.random.seed(0)

    v1 = np.random.rand(n)
    v2 = np.random.rand(n)

    desired_vector_sum = v1 + v2

    np.testing.assert_almost_equal(sim['multiple_vector_sum'],
                                   desired_vector_sum)
    assert sim['multiple_vector_sum'].shape == (n, )

    partials_error_vector_sum = sim.check_partials(
        includes=['comp_multiple_vector_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_vector_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_matrix_random(backend):
    from csdl.examples.valid.ex_sum_multiple_matrix_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    np.random.seed(0)

    M1 = np.random.rand(n * m).reshape((n, m))
    M2 = np.random.rand(n * m).reshape((n, m))

    desired_matrix_sum = M1 + M2

    np.testing.assert_almost_equal(sim['multiple_matrix_sum'],
                                   desired_matrix_sum)
    assert sim['multiple_matrix_sum'].shape == (n, m)

    partials_error_matrix_sum = sim.check_partials(
        includes=['comp_multiple_matrix_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_matrix_sum,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_tensor_random(backend):
    from csdl.examples.valid.ex_sum_multiple_tensor_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    p = 7
    q = 10
    np.random.seed(0)

    T1 = np.random.rand(n * m * p * q).reshape((n, m, p, q))
    T2 = np.random.rand(n * m * p * q).reshape((n, m, p, q))

    desired_tensor_sum = T1 + T2

    np.testing.assert_almost_equal(sim['multiple_tensor_sum'],
                                   desired_tensor_sum)
    assert sim['multiple_tensor_sum'].shape == (n, m, p, q)

    partials_error_tensor_sum = sim.check_partials(
        includes=['comp_multiple_tensor_sum'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_tensor_sum,
                              atol=1.e-5,
                              rtol=1.e-5)


def test_sum_single_matrix_along0_random(backend):
    from csdl.examples.valid.ex_sum_single_matrix_along0_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    np.random.seed(0)

    M1 = np.random.rand(n * m).reshape((n, m))

    desired_single_matrix_sum_axis_0 = np.sum(M1, axis=0)

    np.testing.assert_almost_equal(sim['single_matrix_sum_along_0'],
                                   desired_single_matrix_sum_axis_0)
    assert sim['single_matrix_sum_along_0'].shape == (m, )

    partials_error_single_matrix_axis_0 = sim.check_partials(
        includes=['comp_single_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_single_matrix_axis_0,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_single_matrix_along1_random(backend):
    from csdl.examples.valid.ex_sum_single_matrix_along1_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    np.random.seed(0)

    M1 = np.random.rand(n * m).reshape((n, m))

    desired_single_matrix_sum_axis_1 = np.sum(M1, axis=1)

    np.testing.assert_almost_equal(sim['single_matrix_sum_along_1'],
                                   desired_single_matrix_sum_axis_1)
    assert sim['single_matrix_sum_along_1'].shape == (n, )

    partials_error_single_matrix_axis_1 = sim.check_partials(
        includes=['comp_single_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_single_matrix_axis_1,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_matrix_along0_random(backend):
    from csdl.examples.valid.ex_sum_multiple_matrix_along0_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    np.random.seed(0)

    M1 = np.random.rand(n * m).reshape((n, m))
    M2 = np.random.rand(n * m).reshape((n, m))

    desired_multiple_matrix_sum_axis_0 = np.sum(M1 + M2, axis=0)

    np.testing.assert_almost_equal(sim['multiple_matrix_sum_along_0'],
                                   desired_multiple_matrix_sum_axis_0)

    partials_error_multiple_matrix_axis_0 = sim.check_partials(
        includes=['comp_multiple_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_0,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_multiple_matrix_along1_random(backend):
    from csdl.examples.valid.ex_sum_multiple_matrix_along1_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    n = 3
    m = 6
    np.random.seed(0)

    M1 = np.random.rand(n * m).reshape((n, m))
    M2 = np.random.rand(n * m).reshape((n, m))

    desired_multiple_matrix_sum_axis_1 = np.sum(M1 + M2, axis=1)

    np.testing.assert_almost_equal(sim['multiple_matrix_sum_along_1'],
                                   desired_multiple_matrix_sum_axis_1)

    partials_error_multiple_matrix_axis_1 = sim.check_partials(
        includes=['comp_multiple_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_1,
                              atol=1.e-6,
                              rtol=1.e-6)


def test_sum_concatenate_sums_random(backend):
    from csdl.examples.valid.ex_sum_concatenate_random import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    np.random.seed(0)
    x = np.array(
        [np.sum(np.random.rand(5)),
         np.sum(np.random.rand(4)), 0])

    np.testing.assert_almost_equal(sim['sum_vector'], x)
    np.testing.assert_almost_equal(
        sim['single_vector_sum_1a'],
        sim['single_vector_sum_1a'],
    )

    partials_error_multiple_matrix_axis_1 = sim.check_partials(
        out_stream=None, compact_print=True, method='fd')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_1,
                              atol=1.e-8,
                              rtol=1.e-8)
    partials_error_multiple_matrix_axis_1 = sim.check_partials(
        out_stream=None, compact_print=True, method='cs')
    sim.assert_check_partials(partials_error_multiple_matrix_axis_1,
                              atol=1.e-8,
                              rtol=1.e-8)

import numpy as np
import pytest


def test_integer_index_assignement_overlap(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_integer_overlap import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_integer_index_assignement_out_of_range(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_integer_out_of_range import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_integer_index_assignement(backend):
    from csdl.examples.valid.ex_indices_integer import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    x = np.array([0, 1, 2, 7.4, np.pi, 9, np.pi + 9])
    np.testing.assert_array_equal(sim['x'], x)
    np.testing.assert_array_equal(sim['x0'], x[0])
    np.testing.assert_array_equal(sim['x6'], x[6])
    np.testing.assert_array_equal(sim['x_2'], x[-2])
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_integer_index_integer_reuse(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_integer_reuse import example
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_one_dimensional_index_reuse(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_one_dimensional_reuse import example
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_one_dimensional_index_assignement_overlap(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_one_dimensional_overlap import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_one_dimensional_index_assignement_out_of_range(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_one_dimensional_out_of_range import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_one_dimensional_index_assignement(backend):
    from csdl.examples.valid.ex_indices_one_dimensional import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_array_equal(sim['u'], np.arange(20))
    np.testing.assert_array_equal(sim['v'], np.arange(16))
    np.testing.assert_array_equal(sim['w'], 16 + np.arange(4))
    x = 2 * (np.arange(20) + 1)
    np.testing.assert_array_equal(sim['x'], x)
    np.testing.assert_array_equal(sim['x0_5'], x[0:5])
    np.testing.assert_array_equal(sim['x3_'], x[3:])
    np.testing.assert_array_equal(sim['x2_4'], x[2:4])
    np.testing.assert_array_equal(sim['x_last'], x[-1])
    np.testing.assert_array_equal(sim['z'], x[2])

    y = np.zeros(20)
    y[:16] = 2 * (np.arange(16) + 1)
    y[16:] = 16 + np.arange(4) - 3
    np.testing.assert_array_equal(sim['y'], y)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_multidimensional_dimensional_index_assignement_overlap(
        backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_multidimensional_overlap import example
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_multidimensional_index_assignement_out_of_range(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_indices_multidimensional_out_of_range import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_multidimensional_dimensional_index_assignement(backend):
    from csdl.examples.valid.ex_indices_multidimensional import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_array_equal(
        sim['z'],
        np.arange(6).reshape((2, 3)),
    )
    np.testing.assert_array_equal(
        sim['x'],
        np.arange(6).reshape((2, 3)),
    )
    p = np.arange(30).reshape((5, 2, 3))
    np.testing.assert_array_equal(sim['p'], p)
    # TODO: numpy convention causes shape error without reshape
    np.testing.assert_array_equal(sim['r'], p[0, :, :].reshape(
        (1, 2, 3)))
    np.testing.assert_array_equal(sim['r2'], 5.)
    np.testing.assert_array_equal(
        sim['q'],
        np.arange(30).reshape((5, 2, 3)),
    )
    u = np.arange(20).reshape((1, 20))
    v = 2 * np.arange(20).reshape((1, 20))
    w = np.zeros(40).reshape((2, 20))
    w[0, :] = u
    w[1, :] = v
    np.testing.assert_array_equal(
        sim['s'],
        w,
    )

    t = np.zeros(np.prod((5, 3, 3))).reshape((5, 3, 3))
    t[0:5, 0:-1, 0:3] = np.arange(30).reshape((5, 2, 3))
    np.testing.assert_array_equal(
        sim['t'],
        t,
    )
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_mesh(backend):
    from csdl.examples.valid.ex_indices_mesh import example
    exec('from {} import Simulator'.format(backend))
    nx = 3
    ny = 4
    mesh = np.zeros((nx, ny, 3))
    mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
    mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
    mesh[:, :, 2] = 0.
    sim, rep = example(eval('Simulator'))
    np.testing.assert_array_equal(
        sim['def_mesh'],
        mesh,
    )
    np.testing.assert_array_equal(
        sim['b_pts'],
        mesh[:-1, :, :] * .75 + mesh[1:, :, :] * .25,
    )
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)

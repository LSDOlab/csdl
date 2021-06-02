import numpy as np
import pytest


def test_implicit_nonlinear(backend):
    from csdl.examples.valid.ex_implicit_apply_nonlinear import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    sim['x'] = 1.9
    sim.run()
    np.testing.assert_almost_equal(sim['x'], np.array([1.0]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)

    sim['x'] = 2.1
    sim.run()
    np.testing.assert_almost_equal(sim['x'], np.array([3.0]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_solve_quadratic_bracketed_scalar(backend):
    from csdl.examples.valid.ex_implicit_bracketed_scalar import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['x'], np.array([1.0]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_solve_quadratic_bracketed_array(backend):
    from csdl.examples.valid.ex_implicit_bracketed_array import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['x'],
        np.array([1.0, 3.0]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_implicit_nonlinear_with_subsystems_in_residual(backend):
    from csdl.examples.valid.ex_implicit_with_subsystems import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    # sim.set_val('y', 1.9)
    # sim.run()
    # print(sim['y'])
    np.testing.assert_almost_equal(sim['y'], np.array([1.07440944]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_implicit_nonlinear_with_subsystems_bracketed_scalar(backend):
    from csdl.examples.valid.ex_implicit_with_subsystems_bracketed_scalar import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.07440944]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_implicit_nonlinear_with_subsystems_bracketed_array(backend):
    from csdl.examples.valid.ex_implicit_with_subsystems_bracketed_array import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.07440944, 2.48391993]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)

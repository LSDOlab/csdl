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
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)

    sim['x'] = 2.1
    sim.run()
    np.testing.assert_almost_equal(sim['x'], np.array([3.0]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_fixed_point_iteration(backend):
    from csdl.examples.valid.ex_implicit_fixed_point_iteration import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(
        sim['a'],
        1.1241230297043157,
    )
    np.testing.assert_approx_equal(
        sim['b'],
        1.0798960718178603,
    )
    np.testing.assert_almost_equal(sim['c'], 0.)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_implicit_nonlinear_with_subsystems_in_residual(backend):
    from csdl.examples.valid.ex_implicit_with_subsystems import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_approx_equal(
        sim['a'],
        1.0798960718178603,
    )
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.044583306084130]),
    )

    sim['y'] = 1.9
    sim.run()
    np.testing.assert_approx_equal(
        sim['a'],
        1.0798960718178603,
    )
    np.testing.assert_approx_equal(
        sim['y'],
        np.array([2.659476838580102]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_implicit_multiple_residuals(backend):
    from csdl.examples.valid.ex_implicit_multiple_residuals import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['x'],
        np.array([np.sqrt(3)]),
    )
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


# ----------------------------------------------------------------------


def test_implicit_nonlinear_define_model_inline(backend):
    from csdl.examples.valid.ex_implicit_apply_nonlinear_define_model_inline import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    sim['x'] = 1.9
    sim.run()
    np.testing.assert_almost_equal(sim['x'], np.array([1.0]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)

    sim['x'] = 2.1
    sim.run()
    np.testing.assert_almost_equal(sim['x'], np.array([3.0]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_fixed_point_iteration_define_model_inline(backend):
    from csdl.examples.valid.ex_implicit_fixed_point_iteration_define_model_inline import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(
        sim['a'],
        1.1241230297043157,
    )
    np.testing.assert_approx_equal(
        sim['b'],
        1.0798960718178603,
    )
    np.testing.assert_almost_equal(sim['c'], 0.)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_implicit_nonlinear_with_subsystems_in_residual_define_model_inline(
        backend):
    from csdl.examples.valid.ex_implicit_with_subsystems_define_model_inline import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_approx_equal(
        sim['a'],
        1.0798960718178603,
    )
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.044583306084130]),
    )

    sim['y'] = 1.9
    sim.run()
    np.testing.assert_approx_equal(
        sim['a'],
        1.0798960718178603,
    )
    np.testing.assert_approx_equal(
        sim['y'],
        np.array([2.659476838580102]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_implicit_multiple_residuals_define_model_inline(backend):
    from csdl.examples.valid.ex_implicit_multiple_residuals_define_model_inline import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['x'],
        np.array([np.sqrt(3)]),
    )
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='fd')
    sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)

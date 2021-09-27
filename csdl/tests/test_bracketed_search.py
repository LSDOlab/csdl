import numpy as np
import pytest


def test_solve_quadratic_bracketed_scalar(backend):
    from csdl.examples.valid.ex_bracketed_search_bracketed_scalar import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['x'], np.array([1.0]))

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    # sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_solve_quadratic_bracketed_array(backend):
    from csdl.examples.valid.ex_bracketed_search_bracketed_array import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['x'],
        np.array([1.0, 3.0]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    # sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_bracketed_with_subsystems_scalar(backend):
    from csdl.examples.valid.ex_bracketed_search_with_subsystems_bracketed_scalar import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.04458331]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    # sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)


def test_bracketed_with_subsystems_array(backend):
    from csdl.examples.valid.ex_bracketed_search_with_subsystems_bracketed_array import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(
        sim['y'],
        np.array([1.04458331, 2.65947684]),
    )

    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    # sim.assert_check_partials(result, atol=1.e-6, rtol=1.e-6)

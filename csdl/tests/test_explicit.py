from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_literals(name):
    from csdl.examples.valid.ex_explicit_literals import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(sim['y'], -3.)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_simple_binary(name):
    from csdl.examples.valid.ex_explicit_binary_operations import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(sim['y1'], 7.)
    np.testing.assert_approx_equal(sim['y2'], 5.)
    np.testing.assert_approx_equal(sim['y3'], 1.)
    np.testing.assert_approx_equal(sim['y4'], 6.)
    np.testing.assert_approx_equal(sim['y5'], 2. / 3.)
    np.testing.assert_approx_equal(sim['y6'], 2. / 3.)
    np.testing.assert_approx_equal(sim['y7'], 2. / 3.)
    np.testing.assert_approx_equal(sim['y8'], 9.)
    np.testing.assert_approx_equal(sim['y9'], 4.)
    np.testing.assert_array_almost_equal(sim['y10'], 7 + 2. / 3.)
    np.testing.assert_array_almost_equal(sim['y11'], np.arange(7)**2)
    np.testing.assert_array_almost_equal(sim['y12'], np.arange(7)**2)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_no_registered_outputs(name):
    from csdl.examples.valid.ex_explicit_no_registered_output import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(sim['prod'], 24.)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
    assert len(sim.prob.model._subgroups_myproc) == 1


def test_unary_exprs(name):
    from csdl.examples.valid.ex_explicit_unary import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    x = np.pi
    y = 1
    np.testing.assert_approx_equal(sim['arccos'], np.arccos(y))
    np.testing.assert_approx_equal(sim['arcsin'], np.arcsin(y))
    np.testing.assert_approx_equal(sim['arctan'], np.arctan(x))
    np.testing.assert_approx_equal(sim['cos'], np.cos(x))
    np.testing.assert_approx_equal(sim['cosec'], 1 / np.sin(y))
    np.testing.assert_approx_equal(sim['cosech'], 1 / np.sinh(x))
    np.testing.assert_approx_equal(sim['cosh'], np.cosh(x))
    np.testing.assert_approx_equal(sim['cotan'], 1 / np.tan(y))
    np.testing.assert_approx_equal(sim['cotanh'], 1 / np.tanh(x))
    np.testing.assert_approx_equal(sim['exp'], np.exp(x))
    np.testing.assert_approx_equal(sim['log'], np.log(x))
    np.testing.assert_approx_equal(sim['log10'], np.log10(x))
    np.testing.assert_approx_equal(sim['sec'], 1 / np.cos(x))
    np.testing.assert_approx_equal(sim['sech'], 1 / np.cosh(x))
    np.testing.assert_approx_equal(sim['sin'], np.sin(x))
    np.testing.assert_approx_equal(sim['sinh'], np.sinh(x))
    np.testing.assert_approx_equal(sim['tan'], np.tan(x))
    np.testing.assert_approx_equal(sim['tanh'], np.tanh(x))
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    # assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_explicit_with_subsystems(name):
    from csdl.examples.valid.ex_explicit_with_subsystems import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(sim['x1'], 40.)
    np.testing.assert_approx_equal(sim['x2'], 12.)
    np.testing.assert_approx_equal(sim['y1'], 52.)
    np.testing.assert_approx_equal(sim['y2'], -28.)
    np.testing.assert_approx_equal(sim['y3'], 480.)
    np.testing.assert_approx_equal(sim['prod'], 480.)
    np.testing.assert_approx_equal(sim['y4'], 1600.)
    np.testing.assert_approx_equal(sim['y5'], 144.)
    np.testing.assert_approx_equal(sim['y6'], 196.)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_explicit_cycles(name):
    from csdl.examples.valid.ex_explicit_cycles import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(
        sim['cycle_1.x'],
        1.1241230297043157,
    )
    np.testing.assert_approx_equal(
        sim['cycle_2.x'],
        1.0798960718178603,
    )
    np.testing.assert_almost_equal(sim['cycle_3.x'], 0.)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)

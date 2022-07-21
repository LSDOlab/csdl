import numpy as np
import pytest


def test_exponential_constant_scalar_variable_scalar(backend):
    from csdl.examples.valid.ex_exp_a_exponential_constant_scalar_variable_scalar import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_array_equal(sim['y'], np.array([5**3]))

    result = sim.check_partials(out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)

def test_exponential_constant_scalar_variable_array(backend):
    from csdl.examples.valid.ex_exp_a_exponential_constant_scalar_variable_array import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_array_equal(sim['y'], np.array([5,25,125]))

    result = sim.check_partials(out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)

def test_exponential_constant_array_variable_array(backend):
    from csdl.examples.valid.ex_exp_a_exponential_constant_array_variable_array import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_array_equal(sim['y'], np.array([1,4,27]))

    result = sim.check_partials(out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)

def test_exponential_constant_array_variable_scalar(backend):
    from csdl.examples.invalid.ex_exp_a_exponential_constant_array_variable_scalar import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(ValueError):
        example(eval('Simulator'))


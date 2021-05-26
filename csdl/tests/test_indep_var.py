import numpy as np
import pytest


def test_indep_var(name):
    from csdl.examples.valid.ex_indep_var_simple import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    np.testing.assert_approx_equal(sim['z'], 10.)
    result = sim.check_partials(out_stream=None,
                                compact_print=True,
                                method='cs')
    sim.assert_check_partials(result, atol=1.e-8, rtol=1.e-8)

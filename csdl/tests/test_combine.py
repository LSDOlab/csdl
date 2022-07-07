import numpy as np
import pytest
from csdl.opt.run_all_optimizations import run_all_optimizations

def test_combine_simple(backend):
    from csdl.examples.valid.ex_combine_simple import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    # rep = run_all_optimizations(rep)
    # sim = eval('Simulator(rep)')

    quarter_chord = np.zeros((1, 4, 3))
    quarter_chord[0, :, 0] = 0.5
    quarter_chord[0, :, 1] = np.arange(4)

    np.testing.assert_almost_equal(
        sim['quarter_chord'],
        quarter_chord,
    )
    np.testing.assert_almost_equal(
        sim['widths'],
        1.73205081,
    )

    # TODO: Why does this only work with method='fd'???
    partials_error = sim.check_partials(
        out_stream=None,
        compact_print=True,
        # method='cs',
    )

    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

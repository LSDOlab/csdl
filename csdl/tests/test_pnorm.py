import numpy as np
import pytest


def test_pnorm_axisfree_norm(backend):
    from csdl.examples.valid.ex_pnorm_axis_free import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)
    pnorm_type = 6

    val = np.arange(np.prod(shape)).reshape(shape)

    desired_output = np.linalg.norm(val.flatten(), ord=pnorm_type)
    np.testing.assert_almost_equal(sim['axis_free_pnorm'], desired_output)

    partials_error = sim.check_partials(includes=['comp_axis_free_pnorm'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_pnorm_axiswise(backend):
    from csdl.examples.valid.ex_pnorm_axis_wise import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)
    pnorm_type = 6

    val = np.arange(np.prod(shape)).reshape(shape)

    axis = (1, 3)
    desired_output = np.sum(val**pnorm_type, axis=axis)**(1 / pnorm_type)

    np.testing.assert_almost_equal(sim['axiswise_pnorm'], desired_output)

    partials_error = sim.check_partials(includes=['comp_axiswise_pnorm'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_pnorm_type_not_positive(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_pnorm_type_not_positive import example
    with pytest.raises(Exception):
        example(eval('Simulator'))


def test_pnorm_type_not_even(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_pnorm_type_not_even import example
    with pytest.raises(Exception):
        example(eval('Simulator'))

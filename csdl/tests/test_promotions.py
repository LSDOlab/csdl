
import numpy as np
import pytest


def test_wrong_shape(backend):
    from csdl.examples.invalid.ex_promotions_wrong_shape import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_wrong_shape_two_models(backend):
    from csdl.examples.invalid.ex_promotions_wrong_shape_two_models import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_cycle(backend):
    from csdl.examples.invalid.ex_promotions_cycle import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_cycle_two_models(backend):
    from csdl.examples.invalid.ex_promotions_cycle_two_models import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_inputs(backend):
    from csdl.examples.invalid.ex_promotions_inputs import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_outputs(backend):
    from csdl.examples.invalid.ex_promotions_outputs import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_input_output(backend):
    from csdl.examples.invalid.ex_promotions_input_output import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_promotion(backend):
    from csdl.examples.valid.ex_promotions_promotion import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 7)


def test_partial_promotion(backend):
    from csdl.examples.invalid.ex_promotions_partial_promotion import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_absolute_relative_name(backend):
    from csdl.examples.valid.ex_promotions_absolute_relative_name import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['f'], 4)
    np.testing.assert_almost_equal(sim['model.f'], 4)


def test_same_i_o_unpromoted(backend):
    from csdl.examples.valid.ex_promotions_same_io_unpromoted import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['f'], 1)
    np.testing.assert_almost_equal(sim['model.f'], 2)


def test_two_models_promoted(backend):
    from csdl.examples.invalid.ex_promotions_two_models_promoted import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_two_models_unpromoted(backend):
    from csdl.examples.valid.ex_promotions_two_models_unpromoted import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['f'], 3)
    np.testing.assert_almost_equal(sim['model2.f'], 2)


def test_unconnected_vars(backend):
    from csdl.examples.valid.ex_promotions_unconnected_vars import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 2)


def test_stacked_models(backend):
    from csdl.examples.valid.ex_promotions_stacked_models import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['b'], 11)

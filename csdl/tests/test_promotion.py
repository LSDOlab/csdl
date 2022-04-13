import numpy as np
import pytest

# 8 classes and 8 tests


def test_same_inputs_promoted(backend):
    from csdl.examples.invalid.ex_promotion_same_inputs_promoted import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_same_outputs_promoted(backend):
    from csdl.examples.invalid.ex_promotion_same_outputs_promoted import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_stacked_models(backend):
    from csdl.examples.invalid.ex_promotion_stacked_models import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_two_models_promoted(backend):
    from csdl.examples.invalid.ex_promotion_two_models_promoted import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_same_io_unpromoted(backend):
    from csdl.examples.valid.ex_promotion_same_io_unpromoted import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 2)


def test_unconnected(backend):
    from csdl.examples.valid.ex_promotion_unconnected_vars import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['b'], 2)


def test_two_unpromoted_models(backend):
    from csdl.examples.valid.ex_promotion_two_models_unpromoted import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))


def test_wrong_shape(backend):
    from csdl.examples.valid.ex_promotion_wrong_shape import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 4)

import numpy as np
import pytest


def test_promotion_shape_mismatch(backend):
    from csdl.examples.invalid.ex_promotions_promotion_shape_mismatch import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_wrong_shape_two_models(backend):
    from csdl.examples.invalid.ex_promotions_wrong_shape_two_models import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(ValueError):
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


# TODO: not an error, test values
# def test_input_output(backend):
#     from csdl.examples.invalid.ex_promotions_input_output import example
#     exec('from {} import Simulator'.format(backend))
#     with pytest.raises(KeyError):
#         example(eval('Simulator'))


def test_manual_promotion(backend):
    from csdl.examples.valid.ex_promotions_manual_promotion import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['addition.b'], 1.0)
    np.testing.assert_almost_equal(sim['b'], 7.0)
    np.testing.assert_almost_equal(sim['f'], 4.0)


def test_partial_promotion(backend):
    from csdl.examples.invalid.ex_promotions_partial_promotion import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_absolute_relative_name(backend):
    from csdl.examples.valid.ex_promotions_absolute_relative_name import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['model.f'], 4.0)
    np.testing.assert_almost_equal(sim['f'], 4.0)

def test_same_i_o_unpromoted(backend):
    from csdl.examples.valid.ex_promotions_same_io_unpromoted import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['f'], 1.0)
    np.testing.assert_almost_equal(sim['model.f'], 2.0)


def test_two_models_promoted(backend):
    from csdl.examples.invalid.ex_promotions_two_models_promoted import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_two_models_unpromoted(backend):
    from csdl.examples.valid.ex_promotions_two_models_unpromoted import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['f'], 3.0)
    np.testing.assert_almost_equal(sim['model.f'], 3.0)
    np.testing.assert_almost_equal(sim['model2.f'], 4.0)

def test_unconnected_vars(backend):
    from csdl.examples.valid.ex_promotions_unconnected_vars import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 2.0)

def test_stacked_models(backend):
    from csdl.examples.valid.ex_promotions_stacked_models import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['a'], 3.0)
    np.testing.assert_almost_equal(sim['bmm'], 6.0)
    np.testing.assert_almost_equal(sim['bm'], 8.0)
    np.testing.assert_almost_equal(sim['b'], 11.0)

def test_jumped_promotion(backend):
    from csdl.examples.valid.ex_promotions_jumped_promotion import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['f'], 5.0)


def test_promote_unpromoted(backend):
    from csdl.examples.invalid.ex_promotions_promote_unpromoted import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_promote_nonexistent(backend):
    from csdl.examples.invalid.ex_promotions_promote_nonexistant import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_promote_absolute_name(backend):
    from csdl.examples.invalid.ex_promotions_promote_absolute_name import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_complex(backend):
    from csdl.examples.valid.ex_promotions_complex import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['x3_out'], 1.0)
    np.testing.assert_almost_equal(sim['x4_out'], 10.0)
    np.testing.assert_almost_equal(sim['hierarchical.ModelB.x2'], 6.141)
    np.testing.assert_almost_equal(sim['hierarchical.ModelB.ModelC.x1'],
                                   5.0)
    np.testing.assert_almost_equal(sim['hierarchical.ModelB.ModelC.x2'],
                                   9.0)
    np.testing.assert_almost_equal(sim['hierarchical.ModelF.x3'],
                                   0.01)

def test_parallel_targets(backend):
    from csdl.examples.valid.ex_promotions_parallel_targets import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['f_out'], 15.0)

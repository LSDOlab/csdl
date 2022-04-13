import numpy as np
import pytest

# 19 classes and 19 tests


def test_connections_connect_create_outputs(backend):
    from csdl.examples.invalid.ex_connection_connect_create_outputs import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connections_connect_different_shapes(backend):
    from csdl.examples.invalid.ex_connection_connect_different_shapes import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connect_to_nothing(backend):
    from csdl.examples.invalid.ex_connection_connect_to_nothing import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_cyclical_vars(backend):
    from csdl.examples.invalid.ex_connection_connecting_cyclical_vars import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_two_variables(backend):
    from csdl.examples.invalid.ex_connection_connecting_two_variables import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_unpromoted_names(backend):
    from csdl.examples.invalid.ex_connection_connecting_unpromoted_names import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_vars_in_models(backend):
    from csdl.examples.invalid.ex_connection_connecting_vars_in_models import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_wrong_arg_order(backend):
    from csdl.examples.invalid.ex_connection_connecting_wrong_arg_order import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_two_way_connection(backend):
    from csdl.examples.invalid.ex_connection_two_way_connection import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connect_create_output(backend):
    from csdl.examples.valid.ex_connection_connect_create_outputs import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'][0], 11)
    np.testing.assert_almost_equal(sim['f'][1], 6)


def test_connecting_vars_across_models(backend):
    from csdl.examples.valid.ex_connection_connecting_vars_across_models import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 16)


def test_connection(backend):
    from csdl.examples.valid.ex_connection_connection import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 5)


def test_connection_ignore(backend):
    from csdl.examples.valid.ex_connection_connection_ignore import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 15)


def test_nested_promoted_connections(backend):
    from csdl.examples.valid.ex_connection_nested_promoted_connections import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 8)


def test_nested_unpromoted_connections(backend):
    from csdl.examples.valid.ex_connection_nested_unpromoted_connections import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 8)


def test_nested_unpromoted_connections_variation1(backend):
    from csdl.examples.valid.ex_connection_nested_unpromoted_connections_variation1 import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 8)


def test_nested_unpromoted_connections_variation2(backend):
    from csdl.examples.valid.ex_connection_nested_unpromoted_connections_variation2 import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 8)


def test_unpromoted_connection(backend):
    from csdl.examples.valid.ex_connection_unpromoted_connection import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 5)


def test_value_overwrite_connection(backend):
    from csdl.examples.valid.ex_connection_value_overwrite_connection import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    np.testing.assert_almost_equal(sim['f'], 5)

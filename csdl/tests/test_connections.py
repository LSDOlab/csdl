
import numpy as np
import pytest


def test_unpromoted_connection(backend):
    from csdl.examples.valid.ex_connections_unpromoted_connection import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 3)


def test_connection1(backend):
    from csdl.examples.valid.ex_connections_connection1 import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 3)


def test_nested_promoted_connections(backend):
    from csdl.examples.valid.ex_connections_nested_promoted_connections import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 8)


def test_nested_unpromoted_connections(backend):
    from csdl.examples.valid.ex_connections_nested_unpromoted_connections import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 8)


def test_nested_unpromoted_connections_variation1(backend):
    from csdl.examples.valid.ex_connections_nested_unpromoted_connections_variation1 import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 8)


def test_nested_unpromoted_connections_variation2(backend):
    from csdl.examples.valid.ex_connections_nested_unpromoted_connections_variation2 import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 8)


def test_connecting_wrong_arg_order(backend):
    from csdl.examples.invalid.ex_connections_connecting_wrong_arg_order import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_two_variables(backend):
    from csdl.examples.invalid.ex_connections_connecting_two_variables import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_vars_across_models(backend):
    from csdl.examples.valid.ex_connections_connecting_vars_across_models import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 16)


def test_connecting_vars_in_models(backend):
    from csdl.examples.invalid.ex_connections_connecting_vars_in_models import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_cyclical_vars(backend):
    from csdl.examples.invalid.ex_connections_connecting_cyclical_vars import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connecting_unpromoted_names(backend):
    from csdl.examples.invalid.ex_connections_connecting_unpromoted_names import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_two_way_connection(backend):
    from csdl.examples.invalid.ex_connections_two_way_connection import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_value_overwrite_connection(backend):
    from csdl.examples.valid.ex_connections_value_overwrite_connection import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], 3)


def test_connect_different_shapes(backend):
    from csdl.examples.invalid.ex_connections_connect_different_shapes import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connect_var_to_nothing(backend):
    from csdl.examples.invalid.ex_connections_connect_var_to_nothing import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connect_nothing_to_var(backend):
    from csdl.examples.invalid.ex_connections_connect_nothing_to_var import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_existing_connections(backend):
    from csdl.examples.invalid.ex_connections_existing_connections import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_connect_create_outputs(backend):
    from csdl.examples.valid.ex_connections_connect_create_outputs import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))
    np.testing.assert_almost_equal(sim['y'], np.array([11., 6.]))


def test_error_connect_create_outputs(backend):
    from csdl.examples.invalid.ex_connections_connect_create_outputs import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_false_cycle1(backend):
    from csdl.examples.invalid.ex_connections_false_cycle1 import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))


def test_false_cycle2(backend):
    from csdl.examples.invalid.ex_connections_false_cycle2 import example
    exec('from {} import Simulator'.format(backend))
    with pytest.raises(KeyError):
        example(eval('Simulator'))

from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_unused_inputs_create_no_subsystems(name):
    from openmdao.api import Group
    from csdl.examples.valid.ex_general_unused_inputs import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
    assert sim.prob.model._group_inputs == {}
    assert sim.prob.model._subsystems_allprocs == {}

import numpy as np
import pytest


def test_rotmat_scalar_rotX(backend):
    from csdl.examples.valid.ex_rotmat_scalar_rot_x import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)

    rotmatx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # ROTATION FOR SCALAR
    desired_outputx = rotmatx
    np.testing.assert_almost_equal(sim['scalar_Rot_x'], desired_outputx)

    partials_error = sim.check_partials(includes=['comp_scalar_Rot_x'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_rotmat_scalar_rotY(backend):
    from csdl.examples.valid.ex_rotmat_scalar_rot_y import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)

    rotmaty = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    desired_outputy = rotmaty
    np.testing.assert_almost_equal(sim['scalar_Rot_y'], desired_outputy)


def test_rotmat_same_radian_tensor_rotX(backend):
    from csdl.examples.valid.ex_rotmat_same_radian_tensor_rot_x import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    # Shape of a random tensor rotation matrix
    shape = (2, 3, 4)

    num_elements = np.prod(shape)

    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)

    # Tensor of angles in radians
    rotmatx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    ten_rotmatx = np.tile(rotmatx.flatten(),
                          num_elements).reshape(shape + (3, 3))
    # ROTATION FOR TENSOR

    desired_output = ten_rotmatx
    np.testing.assert_almost_equal(sim['tensor_Rot_x'], desired_output)

    partials_error = sim.check_partials(includes=['comp_tensor_Rot_x'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)


def test_rotmat_same_radian_tensor_rotX(backend):
    from csdl.examples.valid.ex_rotmat_same_radian_tensor_rot_x import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    # Shape of a random tensor rotation matrix
    shape = (2, 3, 4)

    num_elements = np.prod(shape)

    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)

    # Tensor of angles in radians
    rotmatx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    ten_rotmatx = np.tile(rotmatx.flatten(),
                          num_elements).reshape(shape + (3, 3))
    # ROTATION FOR TENSOR

    desired_output = ten_rotmatx
    np.testing.assert_almost_equal(sim['tensor_Rot_x'], desired_output)

    partials_error = sim.check_partials(includes=['comp_tensor_Rot_x'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)


def test_rotmat_diff_radian_tensor_rotX(backend):
    from csdl.examples.valid.ex_rotmat_diff_radian_tensor_rot_x import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    # Shape of a random tensor rotation matrix
    shape = (2, 3, 4)

    num_elements = np.prod(shape)

    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)

    # Tensor of angles in radians
    rotmatx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    ten_rotmatx = np.tile(rotmatx.flatten(),
                          num_elements).reshape(shape + (3, 3))
    # ROTATION FOR TENSOR

    desired_output = ten_rotmatx
    np.testing.assert_almost_equal(sim['tensor_Rot_x'], desired_output)

    partials_error = sim.check_partials(includes=['comp_tensor_Rot_x'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)

import numpy as np

shape1 = (4, )
shape2 = (5, 4)
shape3 = (2, 4, 3)

vec = np.arange(4)
mat = np.arange(20).reshape(shape2)
tens = np.arange(24).reshape(shape3)


def test_einsum_inner_vector_vector(backend):
    from csdl.examples.valid.ex_einsum_old_inner_vector_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i,i->', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_inner1'], desired_output1)

    partials_error1 = sim.check_partials(includes=['comp_einsum_inner1'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)


def test_einsum_inner_tensor_vector(backend):
    from csdl.examples.valid.ex_einsum_old_inner_tensor_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('ijk,j->ik', tens, vec)

    np.testing.assert_array_almost_equal(sim['einsum_inner2'], desired_output2)

    partials_error2 = sim.check_partials(includes=['comp_einsum_inner2'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_outer_vector_vector(backend):
    from csdl.examples.valid.ex_einsum_old_outer_vector_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i,j->ij', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_outer1'], desired_output1)

    # partials_error1 = sim.check_partials(includes=['comp_einsum_outer1'],
    #  out_stream=None,
    #  compact_print=True,
    #  method='cs')
    # sim.assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)


def test_einsum_outer_tensor_vector(backend):
    from csdl.examples.valid.ex_einsum_old_outer_tensor_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('hij,k->hijk', tens, vec)

    np.testing.assert_array_almost_equal(sim['einsum_outer2'], desired_output2)

    partials_error2 = sim.check_partials(includes=['comp_einsum_outer2'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_reorder_matrix(backend):
    from csdl.examples.valid.ex_einsum_old_reorder_matrix import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('ij->ji', mat)

    np.testing.assert_array_almost_equal(sim['einsum_reorder1'],
                                         desired_output1)

    partials_error1 = sim.check_partials(includes=['comp_einsum_reorder1'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)


def test_einsum_reorder_tensor(backend):
    from csdl.examples.valid.ex_einsum_old_reorder_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('ijk->kji', tens)

    np.testing.assert_array_almost_equal(sim['einsum_reorder2'],
                                         desired_output2)

    partials_error2 = sim.check_partials(includes=['comp_einsum_reorder2'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_vector_summation(backend):
    from csdl.examples.valid.ex_einsum_old_vector_summation import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i->', vec)

    np.testing.assert_array_almost_equal(sim['einsum_summ1'], desired_output1)

    partials_error1 = sim.check_partials(includes=['comp_einsum_summ1'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)


def test_einsum_tensor_summation(backend):
    from csdl.examples.valid.ex_einsum_old_tensor_summation import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('ijk->', tens)

    np.testing.assert_array_almost_equal(sim['einsum_summ2'], desired_output2)

    partials_error2 = sim.check_partials(includes=['comp_einsum_summ2'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_multiplication_sum(backend):
    from csdl.examples.valid.ex_einsum_old_multiplication_sum import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i,j->j', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_special1'],
                                         desired_output1)

    partials_error1 = sim.check_partials(includes=['comp_einsum_special1'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)


def test_einsum_special(backend):
    from csdl.examples.valid.ex_einsum_old_multiple_vector_sum import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('i,j->', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_special2'],
                                         desired_output2)

    partials_error2 = sim.check_partials(includes=['comp_einsum_special2'],
                                         out_stream=None,
                                         compact_print=True,
                                         method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_inner_vector_vector(backend):
    from csdl.examples.valid.ex_einsum_old_inner_vector_vector_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i,i->', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_inner1_sparse_derivs'],
                                         desired_output1)

    partials_error1 = sim.check_partials(
        includes=['comp_einsum_inner1_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_inner_tensor_vector(backend):
    from csdl.examples.valid.ex_einsum_old_inner_tensor_vector_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('ijk,j->ik', tens, vec)

    np.testing.assert_array_almost_equal(sim['einsum_inner2_sparse_derivs'],
                                         desired_output2)

    partials_error2 = sim.check_partials(
        includes=['comp_einsum_inner2_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_outer_vector_vector(backend):
    from csdl.examples.valid.ex_einsum_old_outer_vector_vector_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i,j->ij', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_outer1_sparse_derivs'],
                                         desired_output1)

    partials_error1 = sim.check_partials(
        includes=['comp_einsum_outer1_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_outer(backend):
    from csdl.examples.valid.ex_einsum_old_outer_tensor_vector_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('hij,k->hijk', tens, vec)

    np.testing.assert_array_almost_equal(sim['einsum_outer2_sparse_derivs'],
                                         desired_output2)

    partials_error2 = sim.check_partials(
        includes=['comp_einsum_outer2_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_reorder_matrix(backend):
    from csdl.examples.valid.ex_einsum_old_reorder_matrix_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('ij->ji', mat)

    np.testing.assert_array_almost_equal(sim['einsum_reorder1_sparse_derivs'],
                                         desired_output1)

    partials_error1 = sim.check_partials(
        includes=['comp_einsum_reorder1_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)


def test_einsum_sparse_reorder(backend):
    from csdl.examples.valid.ex_einsum_old_reorder_tensor_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('ijk->kji', tens)

    np.testing.assert_array_almost_equal(sim['einsum_reorder2_sparse_derivs'],
                                         desired_output2)

    partials_error2 = sim.check_partials(
        includes=['comp_einsum_reorder2_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_sparse_vector_summation(backend):
    from csdl.examples.valid.ex_einsum_old_vector_summation_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i->', vec)

    np.testing.assert_array_almost_equal(sim['einsum_summ1_sparse_derivs'],
                                         desired_output1)

    partials_error1 = sim.check_partials(
        includes=['comp_einsum_summ1_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)


def test_einsum_sparse_tensor_summation(backend):
    from csdl.examples.valid.ex_einsum_old_tensor_summation_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('ijk->', tens)

    np.testing.assert_array_almost_equal(sim['einsum_summ2_sparse_derivs'],
                                         desired_output2)

    partials_error2 = sim.check_partials(
        includes=['comp_einsum_summ2_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_sparse_multiplication_sum(backend):
    from csdl.examples.valid.ex_einsum_old_multiplication_sum_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output1 = np.einsum('i,j->j', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_special1_sparse_derivs'],
                                         desired_output1)

    partials_error1 = sim.check_partials(
        includes=['comp_einsum_special1_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_special(backend):
    from csdl.examples.valid.ex_einsum_old_multiple_vector_sum_sparse import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    desired_output2 = np.einsum('i,j->', vec, vec)

    np.testing.assert_array_almost_equal(sim['einsum_special2_sparse_derivs'],
                                         desired_output2)

    partials_error2 = sim.check_partials(
        includes=['comp_einsum_special2_sparse_derivs'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)

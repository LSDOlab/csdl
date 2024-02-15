from csdl.lang.variable import Variable
from csdl.lang.output import Output
import csdl.operations as ops
import numpy as np
from csdl.solvers.linear.direct import DirectSolver
from scipy.sparse import issparse

def solve(A: Variable, b: Variable, solver = DirectSolver()):
    '''
    This function solves a linear system.

    **Parameters**

    A: Variable
        2D nxn CSDL Variable/scipy sparse matrix/np.ndarray

    b: tuple[int]
        2D/1D nx1 CSDL Variable/np.ndarray
    '''
    
    # ================================ CHECKS ================================
    # check types for A and b
    if not isinstance(A, Variable) and not issparse(A) and not isinstance(A, np.ndarray):
        raise TypeError(f"A in csdl.solve is not an Variable/sparse matrix/np.ndarray object")
    if not isinstance(b, Variable) and not isinstance(b, np.ndarray):
        raise TypeError(f"b in csdl.solve is not an Variable/np.ndarray object")

    # check dimensions/shapes for A and b
    if len(A.shape) != 2:
        raise ValueError(f"A in csdl.solve must be a 2D array. {A.shape} given")
    
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError(f"A in csdl.solve must be a square array. {A.shape} given")

    if b.shape[0] != n:
        raise ValueError(f"b in csdl.solve must have the same number of rows as A ({n}). {b.shape} given")
    if len(b.shape) == 2:
        m = b.shape[1]
    elif len(b.shape) == 1:
        m = 1
    else:
        raise ValueError(f"b in csdl.solve must be a 2D/1D array. {b.shape} given")
    if m != 1:
        raise ValueError(f"b in csdl.solve cannot have more than one column. {b.shape} given")  
    
    # check to make sure atleast one of A and b is a CSDL variable
    if not isinstance(A, Variable) and not isinstance(b, Variable):
        raise TypeError(f"Atleast one of A and b in csdl.solve must be a CSDL Variable object. {type(A)} and {type(b)} given")
    
    # check to make sure that the solver is a solver object
    from csdl.solvers.linear_solver import LinearSolver
    if not isinstance(solver, LinearSolver):
        raise TypeError(f"solver in csdl.solve must be a LinearSolver object. {solver} given")
    # ================================ CHECKS ================================

    # valid inputs from here on out.
    # preprocessing:
    from csdl.operations.solve_linear import SolveLinear
    out_shape = b.shape
    b_reshaped = b.reshape((n, 1))

    # Finally, create the operation
    op = SolveLinear(A = A, b = b_reshaped, n=n, solver = solver)
    op.outs = (Output(
        None,
        op=op,
        shape=(n, 1),
    ), )
    return op.outs[0].reshape(out_shape)


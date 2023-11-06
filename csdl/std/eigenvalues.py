import csdl.operations as ops
from csdl.lang.variable import Variable
from csdl.lang.output import Output
import warnings


def eigenvalues(var):
    """
    Computes the eigenvalues of a square matrix
    
    Parameters
    ----------
    var : Variable
        A 2D square matrix

    Returns
    -------
    Tuple[variable, variable]
        A tuple of two variables, the first of which is the real part of the eigenvalues, the second of which is the imaginary part of the eigenvalues.
        The eigenvalues are returned in the sorted from the largest to the smallest real part. This may cause issues when verifying derivatives with 
        finite differences as the order of the eigenvalues may change during the perturbation.
    """

    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    
    if len(var.shape) != 2:
        raise ValueError(f"csdl.eigenvalues expects a 2D square matrix. {var.shape}.given")
    else:
        if var.shape[0] != var.shape[1]:
            raise ValueError(f"csdl.eigenvalues expects a 2D square matrix. {var.shape}.given")


        if var.shape[0] > 100:
            # warning that says eigenvalues is slow for large matrices due to inversion.
            warnings.warn(f"csdl.eigenvalues is currently inefficient for large matrices.")
        
    # No more errors
    n = var.shape[0]
    op = ops.eigenvalues(var, n = n)
    op.outs = (
        Output(
            None,
            op=op,
            shape=(n,),
        ), 
        Output(
            None,
            op=op,
            shape=(n,),
        )
        )

    return op.outs

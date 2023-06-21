from csdl.lang.variable import Variable
from csdl.utils.get_shape_val import get_shape_val

class DeclaredVariable(Variable):
    """
    Base class for outputs; used to prevent circular imports
    """
    def __init__(
        self,
        name,
        val=1.0,
        shape=(1, ),
        src_indices=None,
        flat_src_indices=None,
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
        op=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            name,
            val=val,
            shape=shape,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            distributed=distributed,
            op=op,
            *args,
            **kwargs,
        )
        self.src_indices = src_indices
        self.flat_src_indices = flat_src_indices
        self.shape, self.val = get_shape_val(shape, val)


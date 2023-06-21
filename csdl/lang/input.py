from csdl.lang.variable import Variable
from csdl.utils.get_shape_val import get_shape_val


class Input(Variable):
    """
    Container for creating a value that is constant during model
    evaluation; i.e. independent variable, or design variable
    """
    def __init__(
        self,
        name: str,
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
        *args,
        **kwargs,
    ):
        super().__init__(
            name,
            val=val,
            shape=shape,
            src_indices=src_indices,
            flat_src_indices=flat_src_indices,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            distributed=distributed,
            *args,
            **kwargs,
        )

        self.shape, self.val = get_shape_val(shape, val)
        # if name == 'mtx':
        #     import numpy as np
        #     print(np.linalg.norm(self.val))
        #     exit()
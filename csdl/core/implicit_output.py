from csdl.core.output import Output


class ImplicitOutput(Output):
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
        res_units=None,
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
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
            op=op,
            *args,
            **kwargs,
        )
        self.res_units = res_units
        self.lower = lower
        self.upper = upper
        self.ref = ref
        self.ref0 = ref0
        self.res_ref = res_ref

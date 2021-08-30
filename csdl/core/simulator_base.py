class _ReprClass(object):
    """
    Class for defining objects with a simple constant string __repr__.

    This is useful for constants used in arg lists when you want them to appear in
    automatically generated source documentation as a certain string instead of python's
    default representation.
    """
    def __init__(self, repr_string):
        """
        Inititialize the __repr__ string.

        Parameters
        ----------
        repr_string : str
            The string to be returned by __repr__
        """
        self._repr_string = repr_string

    def __repr__(self):
        """
        Return our _repr_string.

        Returns
        -------
        str
            Whatever string we were initialized with.
        """
        return self._repr_string


# Use this as a special value to be able to tell if the caller set a value for the optional
# out_stream argument. We run into problems running testflo if we use a default of sys.stdout.
_DEFAULT_OUT_STREAM = _ReprClass("DEFAULT_OUT_STREAM")

msg = "The CSDL Simulator base class does not implement any methods. Please install a CSDL compiler backend with a Simulator class that conforms to the SDL Simulator API."


class SimulatorBase:
    """
    A class that can be used as a base class for the ``Simulator`` class
    that a CSDL compiler backend would provide.
    This class is only here so that CSDL users and CSDL compiler backend
    developers have API documentation.
    CSDL users are not to use the ``SimulatorBase`` class provided by
    ``csdl``, only the ``Simulator`` class provided by the CSDL compiler
    backend of choice.
    """
    def __init__(self, model, reorder=False):
        raise NotImplementedError(msg)

    def __getitem__(self, key):
        raise NotImplementedError(msg)

    def __setitem__(self, key, val):
        raise NotImplementedError(msg)

    def run(self):
        raise NotImplementedError(msg)

    def visualize_model(self):
        raise NotImplementedError(msg)

    def check_partials(
        self,
        out_stream=_DEFAULT_OUT_STREAM,
        includes=None,
        excludes=None,
        compact_print=False,
        abs_err_tol=1e-6,
        rel_err_tol=1e-6,
        method='fd',
        step=None,
        form='forward',
        step_calc='abs',
        force_dense=True,
        show_only_incorrect=False,
    ):
        raise NotImplementedError(msg)

    def assert_check_partials(self, result, atol=1e-8, rtol=1e-8):
        raise NotImplementedError(msg)

try:
    from csdl.lang.model import Model
except ImportError:
    pass
from csdl.lang.concatenation import Concatenation
from warnings import warn
from typing import Set


def define_models_recursively(model: 'Model'):
    """
    Construct intermediate representation for each model in the
    hierarchy from the top down
    """
    if model.defined is False:
        # this does nothing if user defines model inline; whether or not
        # user defines model inline, submodels are not defined until
        # this context
        model.define()
        model.defined = True
        # Models must be defined from the top down because ubgraphs are
        # not added to the model until after the model is defined
        for s in model.subgraphs:
            m = s.submodel
            define_models_recursively(m)

        # check for empty model
        if model.registered_outputs == [] and model.subgraphs == []:
            if model.inputs == []:
                raise ValueError(
                    "This model doesn't do anything. Models must register inputs, outputs, or contain submodels."
                )
            else:
                warn(
                    "This model only registers inputs for the main model"
                )

        # Check if all design variables are inputs
        input_names: Set[str] = {inp.name for inp in model.inputs}
        for name in model.design_variables.keys():
            if name not in input_names:
                raise KeyError(
                    "{} is not the CSDL name of an input to the model".
                    format(name))

        # Check that all outputs are defined
        # for output in self.sorted_nodes:
        for output in model.registered_outputs:
            if isinstance(output, Concatenation):
                if output.defined is False:
                    raise ValueError(
                        "Output not defined for {}. When defining a concatenation, at least one index must be defined in terms of another CSDL Variable"
                        .format(repr(output)))
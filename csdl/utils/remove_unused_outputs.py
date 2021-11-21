from typing import List


def remove_unused_outputs(
    model,
    residuals: List[str],
    expose: List[str],
):
    """
    Use when a `Model` defines residuals for an `ImplicitOperation`.
    Remove nodes in `model.registered_outputs` that are not
    residuals or exposed intermediate outputs, leaving only nodes of `Output` type that define a
    residual and not an output to be used by a submodel.

    **Parameters**

    `model: Model`

        `Model` that defines residuals

    `residuals: List[str]`

        List of names of registered outputs designated as residuals

    `expose: List[str]`

        List of names of intermediate outputs designated for exposure to
        parent `Model` of the `ImplicitOperation`
    """

    # remove unused outputs from registered outputs, keep residuals and
    # exposed intermediate outputs
    remove = []
    for var in model.registered_outputs:
        if var.name not in residuals + expose:
            remove.append(var.name)
    for rem in remove:
        model.registered_outputs.remove(rem)

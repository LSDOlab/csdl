from csdl.core.subgraph import Subgraph
from csdl.core.variable import Variable
from typing import List, Set

# def find_possible_promotions(
#     subgraph: Subgraph,
#     ports: str,
# )->Set[Variable]:
#     # TODO: return list of variables, or strings?
#     if ports == 'inputs':
#         possible_promotions = set(subgraph.inputs + subgraph.promoted_inputs)
#     elif ports == 'outputs':
#         possible_promotions = set(subgraph.outputs + subgraph.promoted_outputs)
#     elif ports == 'declared':
#         possible_promotions = set(subgraph.declared_variables + subgraph.promoted_declared_variables)
#     else:
#         raise ValueError("Invalid option for ports, {}".format(ports))

#     # TODO: include only user-specified promotions

#     return possible_promotions

# # TODO: Subgraphs can no longer be registered outputs; if they
# # don't have any dependencies, then they're added at the end of
# # sorted_nodes
# def find_promoted_inputs(subgraph: Subgraph) -> List[str]:
#     """
#     return names of declared variables promoted to parent level
#     """
#     promoted_inputs = find_possible_promotions(subgraph, 'inputs')
#     # outputs in submodel contained in subgraph will not be promoted
#     if subgraph.promotes is [] or subgraph.promotes_inputs is []:
#         return []

#     # outputs in submodel contained in subgraph will be promoted
#     # valid promotions checked after this function returns
#     promoted_ins = []
#     if subgraph.promotes or '*' in subgraph.promotes_inputs:
#     if subgraph.promotes_inputs is None:

#         return [
#             var.name for var in subgraph.submodel.inputs
#         ] + subgraph.promoted_ins

#     promoted_ins = []
#     promotion_candidates = set(
#         [var.name for var in subgraph.submodel.declared_variables] +
#         subgraph.promoted_ins)
#     # check that user is attempting to promote a variable that exists
#     for name in subgraph.promotes_inputs:
#         if name not in promotion_candidates:
#             raise ValueError(
#                 "{} is not a valid declared variable of {} or promoted declared variable of {}"
#                 .format(name, subgraph.name, subgraph.name))

#     # gather inputs/outputs to promote to parent model
#     for name in subgraph.promotes + subgraph.promotes_inputs:
#         if name in promotion_candidates:
#             promoted_ins.append(name)
#     return promoted_ins

# def find_promoted_outputs(subgraph: Subgraph):
#     """
#     return names of inputs and outputs promoted to parent level
#     """
#     # outputs in submodel contained in subgraph will not be promoted
#     if subgraph.promotes is [] or subgraph.promotes_outputs is []:
#         return []

#     # outputs in submodel contained in subgraph will be promoted
#     # valid promotions checked after this function returns
#     if '*' in subgraph.promotes or '*' in subgraph.promotes_outputs:
#         # TODO: Subgraphs are no longer registered outputs; if they
#         # don't have any dependencies, then they're added at the end of
#         # sorted_nodes
#         return [
#             var.name for var in subgraph.submodel.registered_outputs +
#             subgraph.submodel.inputs
#         ] + subgraph.promoted_outs

#     promoted_outs = []
#     promotion_candidates = [
#         var.name for var in subgraph.submodel.registered_outputs +
#         subgraph.submodel.inputs
#     ] + subgraph.promoted_outs
#     # check that user is attempting to promote a variable that exists
#     for name in subgraph.promotes_outputs:
#         if name not in promotion_candidates:
#             raise ValueError(
#                 "{} is not a valid input/output of {} or promoted input/output of {}"
#                 .format(name, subgraph.name, subgraph.name))

#     # gather inputs/outputs to promote to parent model
#     for name in subgraph.promotes + subgraph.promotes_outputs:
#         if name in promotion_candidates:
#             promoted_outs.append(name)
#     return promoted_outs

try:
    from csdl.lang.model import Model
except ImportError:
    pass
from typing import List, Tuple


def issue_user_specified_connections(model: 'Model'):
    """
    Issue connections after all promotions have been resolved and
    connections formed due to promotions have been issued. User allowed
    to specify connections using relative promoted or relative
    unpromoted names. Stores connections in model using relative
    promoted names only.
    """
    for s in model.subgraphs:
        m = s.submodel
        issue_user_specified_connections(m)
        promoted_user_declared_connections: List[Tuple[str, str]] = []
        for (a, b) in m.user_declared_connections:
            if a in m.promoted_to_unpromoted.keys():
                promoted_a = a
            elif a in m.unpromoted_to_promoted.keys():
                promoted_a = m.unpromoted_to_promoted[a]
            else:
                raise KeyError(
                    "Variable {} is not a valid source (input or output) for conenction."
                    .format(a))
            if b in m.promoted_to_unpromoted.keys():
                promoted_b = b
            elif b in m.unpromoted_to_promoted.keys():
                promoted_b = m.unpromoted_to_promoted[b]
            else:
                raise KeyError(
                    "Variable {} is not a valid sink (declared variable) for conenction."
                    .format(a))

            promoted_user_declared_connections.append(
                (promoted_a, promoted_b))

        m.connections = list(
            set(m.connections + promoted_user_declared_connections))

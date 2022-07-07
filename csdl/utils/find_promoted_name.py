from typing import Dict, Set


def find_promoted_name(
    name: str,
    promoted_to_unpromoted: Dict[str, Set[str]],
    unpromoted_to_promoted: Dict[str, str],
) -> str:
    if name in promoted_to_unpromoted.keys():
        return name
    return unpromoted_to_promoted[name]

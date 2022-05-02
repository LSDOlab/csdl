from csdl.lang.standard_operation import StandardOperation
from typing import Any


def check_property(op: StandardOperation, prop: Any, status: bool):
    try:
        check = op.properties[prop] == status
    except:
        check = False
    return check

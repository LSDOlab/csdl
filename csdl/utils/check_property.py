def check_property(op, prop, status):
    try:
        check = op.properties[prop] == status
    except:
        check = False
    return check

def check_property(b, op, prop, truthy):
    try:
        b = b and op.properties[prop] == truthy
    except:
        pass
    return b

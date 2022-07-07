def prepend_namespace(namespace: str, name: str) -> str:
    return name if namespace == '' else namespace + '.' + name


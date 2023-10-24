

def check_banned_chars(string):
    """
    Check if string contains banned characters.
    All characters is string must be valid for python variable names 
    """

    error_str = f"Variable/Model name {string} contains invalid characters. Only alphanumeric characters or underscores are allowed."
    for char in string:
       if not (char.isalnum() or char == '_'):
          raise ValueError(error_str)
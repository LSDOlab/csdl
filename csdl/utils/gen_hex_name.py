# https://slavik.meltser.info/convert-base-10-to-base-64-and-vise-versa-using-javascript/
def gen_hex_name(num: int) -> str:
    """
    Convert an integer from base 10 to base 64 (string)

    Parameters
    ----------
    num: int
        Number to convert, in base 10

    Returns
    -------
    str
        4-digit number in base 63, with leading zeros
    """
    # All characters that OpenMDAO allows in Group names
    char_set = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    base = len(char_set)
    string_rep = ""
    remainder = 0

    # name will have leading zeros
    prefix = "_0000"
    terminate = False
    while terminate == False:
        remainder = num % base
        num -= remainder
        num = int(num / base)
        string_rep = char_set[remainder] + string_rep
        prefix = prefix[:-1]
        if num == 0:
            terminate = True

    return prefix + string_rep

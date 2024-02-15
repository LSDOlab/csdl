def format_number(number):
    if number == 0.0:
        s = f'{number}'
    elif abs(number) < 1:
        s =  "{:.2e}".format(number)
    elif abs(number) < 10:
        s =  "{:.3f}".format(number)
    else:
        s =  "{:.2e}".format(number)
    s = s.replace("e-0", "e-")
    s = s.replace("e-", "e-")
    s = s.replace("e+0", "e")
    s = s.replace("e+", "e")
    return s


if __name__ == "__main__":
    # Test the function with example numbers
    number1 = 1.1234567788
    number2 = 123456789
    number3 = 1234567899999
    number4 = 0.00000000000000123456789
    number5 = 0.0

    formatted_number1 = format_number(number1)
    formatted_number2 = format_number(number2)
    formatted_number3 = format_number(number3)
    formatted_number4 = format_number(number4)
    formatted_number5 = format_number(number5)

    print(formatted_number1)
    print(formatted_number2)
    print(formatted_number3)
    print(formatted_number4)
    print(formatted_number5)
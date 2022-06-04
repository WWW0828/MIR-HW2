def str2float(str2convert):
    if str2convert[-1] in '\n ':
        convert_str = str2convert[:-1]
    else:
        convert_str = str2convert
    num = 0
    after_point = 0
    for i in convert_str:
        if(i == '.'):
            after_point = 1
            continue
        if(after_point == 0):
            num = num * 10 + int(i)
        else:
            num = num + int(i) * 10**(-1 * after_point)
            after_point += 1
    return num
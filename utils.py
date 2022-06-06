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

def P_SCORE(estimate, weight, reference):
    hit1, hit2 = 0, 0
    if abs(estimate[0] - reference)/reference <= 0.08:
        hit1 = 1
    if abs(estimate[1] - reference)/reference <= 0.08:
        hit2 = 1
    return weight * hit1 + (1 - weight) * hit2

def ALOTC_SCORE(estimate, reference):
    hit1, hit2 = 0, 0
    if abs(estimate[0] - reference)/reference <= 0.08:
        hit1 = 1
    if abs(estimate[1] - reference)/reference <= 0.08:
        hit2 = 1
    if hit1 or hit2:
        return 1
    return 0

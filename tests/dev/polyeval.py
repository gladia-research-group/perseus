def horner(coeffs, x):
    r = coeffs[-1]
    for c in coeffs[-2::-1]:
        r = r * x + c
    return r

def pws_storage_peterson_stockmeyer(coeffs, x):
    i = 0
    curr_x = x
    pows = []
    pows_levels = []
    while 2 ** i < len(coeffs):
        pows.append(curr_x)
        pows_levels.append(i)
        curr_x = curr_x * curr_x
        i += 1

    r = 0
    for i, c in enumerate(coeffs):
        curr_x = None
        curr_pow_idx = 0
        while i > 0:
            if i % 2 == 1:
                if curr_x is None:
                    curr_x = c * pows[curr_pow_idx]
                else:
                    curr_x = curr_x * pows[curr_pow_idx]
            curr_pow_idx += 1
            i = i // 2

        if curr_x is None:
            r += c
        else:
            r += curr_x
    return r

def computation_only_peterson_stockmeyer(coeffs, x):
    r = 0
    for i, c in enumerate(coeffs):
        curr_x = None
        running_x = x
        curr_pow_idx = 0
        while i > 0:
            if i % 2 == 1:
                if curr_x is None:
                    curr_x = c * running_x
                else:
                    curr_x = curr_x * running_x
            running_x = running_x * running_x
            curr_pow_idx += 1
            i = i // 2
        if curr_x is None:
            r += c
        else:
            r += curr_x
    return r
        

if __name__ == "__main__":
    coeffs = [-241.57, 6574.1123, 3, 4, 5, 6, 7.3, 8, 9, 10]
    x = 1.1235
    print(horner(coeffs, x))
    print(pws_storage_peterson_stockmeyer(coeffs, x))
    print(computation_only_peterson_stockmeyer(coeffs, x))
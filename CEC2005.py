import numpy as np

def f1(*args):
    # f1: Shifted Sphere (optimum = -450)
    result = -450

    for x in args:
        result += np.power(x, 2)

    return result

def f2(*args):
    # f2: Shifted Schwefel's Problem 1.2 (optimum = -450)
    result = -450

    for i in range(len(args)):

        cumsum = 0

        for j in range(i):

            cumsum += args[j]

        result += np.power(cumsum, 2)

    return result

def f3(*args):
    #f3: Shifted Rotated High Conditioned Elliptic Function

    D = len(args)

    result = -450

    for i in range(len(args)):
        result += np.power(1000000, i/D-1) * np.power(args[i], 2) # here, i starts from zero

    return result

import numpy as np

a_param = 1e3
b_param = 1
c_param = 0.1

def f(t, z, a=a_param, b=b_param, c=c_param):
    x, y = z
    dx = a * (-(x**3 / 3.0 - x) + y)
    dy = -x - b * y + c
    return np.array([dx, dy], dtype=float)

def jacobian(t, z, a=a_param, b=b_param, c=c_param):
    x, y = z
    return np.array([
        [a * (1.0 - x**2), a],
        [-1.0,             -b]
    ], dtype=float)
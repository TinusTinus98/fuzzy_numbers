import numpy as np


def generate(x, y):
    assert len(x) == len(y)
    n = len(x)
    x1 = np.repeat(x, n)
    x2 = np.resize(np.repeat(np.array([y]), axis=0), (n**2))
    y1 = np.repeat(y, n)
    y2 = np.resize(np.repeat(np.array([y]), axis=0), (n**2))
    ns = np.sign(x2 - x1) == np.sign(y2 - x1)
    nd = n - ns
    return (ns - nd) / (ns + nd)

def p_value():
    return 0

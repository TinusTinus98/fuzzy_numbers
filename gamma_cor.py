import numpy as np


def generate(x, y):
    assert len(x) == len(y)
    n = len(x)
    x1 = np.repeat(x, n)
    x2 = np.resize(np.repeat(np.array([y]),n, axis=0), (n**2))
    y1 = np.repeat(y, n)
    y2 = np.resize(np.repeat(np.array([y]),n, axis=0), (n**2))
    ns = np.sum(np.sign(x2 - x1) == np.sign(y2 - y1))
    nd = n**2 - ns
    return (ns - nd) / (ns + nd)
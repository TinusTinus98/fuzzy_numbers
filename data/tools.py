import numpy as np


def k_cross_partition(n, k):
    array = [i for i in range(n)]
    shuffled = np.random.permutation(array)
    splitted_list = np.array_split(shuffled, k)
    return splitted_list


print(k_cross_partition(100, 3))

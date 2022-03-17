"""
This script is trying to sort 3D coordinates in a spatial decomposition way.
"""
from itertools import chain
import numpy as np
from math import floor
import time
import matplotlib.pyplot as plt
np.random.seed(314)

N = 10000  # number of coordinates/particles

# 1D
"""
Obviously, we can just sort the 1D coordinates and get a spatical decomposition.

This won't be discussed.
"""


# 2D
"""
Algorithm:
    1) sort the the 2D coordinates by the first dimension
    2) divide the coordinates into two
    3) for each half, sort the coordinates by the second dimension
    4) for each sorted half, divide the coordinates into two, and for each half in halves, sort by the first dimension
    5) repeat until all sorted.
"""


def sort_divide(coords, dim):
    """
    Sort an array 'coords' by its dimension 'dim', and then divide it into two halves.

    params:
        coords: numpy.ndarray-like, array to sort and divide
        dim: int, the dimension to sort

    returns:
        coords_1: first half of the sorted coords
        coords_2: second half of the sorted coords
    """
    N, d = np.shape(coords)
    if N == 1:
        return coords[0] 
    dim_other = 1 if dim == 0 else 0
    coords = coords[coords[:, dim].argsort()]
    half_index = floor(N/2)
    return sort_divide(coords[0:half_index,:], dim_other), sort_divide(coords[half_index:N,:], dim_other)

def flatten(data):
    result = []
    for item in data:
        if isinstance(item, tuple):
            result.extend(flatten(item))
        else:
            result.append(item)
    return tuple(result)

t = np.arange(N)
a = np.random.rand(N, 2)
a = sort_divide(a, 0)
a = np.stack(flatten(a))
plt.scatter(a[:, 0], a[:, 1], c=t, cmap='jet')
plt.tight_layout()
plt.savefig('2D.png', dpi=300)


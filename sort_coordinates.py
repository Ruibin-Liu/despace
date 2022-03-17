"""
This script is trying to sort 3D coordinates in a spatial decomposition way.
"""

import sys
from itertools import chain
import numpy as np
from math import floor
import matplotlib.pyplot as plt
np.random.seed(314)

N = int(sys.argv[1])  # number of coordinates/particles
d = 2
if len(sys.argv) >= 3:
    d = int(sys.argv[2])  # number of dimensions

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
    if d == 2:
        dim_next = 0 if dim == 1 else 1
    elif d == 3:
        if dim == 0:
            dim_next = 1
        elif dim == 1:
            dim_next = 2
        else:
            dim_next = 0
    else:
        raise ValueError("Only 2D and 3D have been implemented.")
    coords = coords[coords[:, dim].argsort()]
    half_index = floor(N/2)
    return sort_divide(coords[0:half_index,:], dim_next), sort_divide(coords[half_index:N,:], dim_next)

def flatten(data):
    result = []
    for item in data:
        if isinstance(item, tuple):
            result.extend(flatten(item))
        else:
            result.append(item)
    return tuple(result)

t = np.arange(N)
a = np.random.rand(N, d)
a = sort_divide(a, 0)
a = np.stack(flatten(a))
fig = plt.figure(figsize=(3.5, 3.5))
if d == 2:
    ax = fig.add_subplot()
    plt.scatter(a[:, 0], a[:, 1], c=t, cmap='jet')
    ax.set_xlim([0.0, 1.0])
    ax.axis('equal')
    plt.tight_layout()
elif d == 3:
    ax = fig.add_subplot(projection='3d')
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], c=t, cmap='jet')
else:
    exit()
plt.savefig(f'{d}D_{N}.png', dpi=300)


"""
This script is trying to sort 3D coordinates in a spatial decomposition way.
"""

import sys
from itertools import chain
from urllib.parse import _NetlocResultMixinStr
import numpy as np
from math import floor
import matplotlib.pyplot as plt

N = int(sys.argv[1])  # number of coordinates/particles
d = 2
if len(sys.argv) >= 3:
    d = int(sys.argv[2])  # number of dimensions
show_arrow = True  # whether to show the arrows connecting two adjacent data points


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

def next_dim(current_dim, d):
    """
    Generate the next dimension from the current dimension and the number of dimensions

    params:
        current_dim: int, the current dimension in an array-like object
        d: int, the number of dimensions

    returns:
        dim_next: int, the next dimension in a circular way
    """
    return current_dim + 1 if current_dim + 1 < d else 0

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
    dim_next = next_dim(dim, d)
    coords = coords[coords[:, dim].argsort()]
    half_index = floor(N/2)
    return sort_divide(coords[0:half_index,:], dim_next), sort_divide(coords[half_index:N,:], dim_next)

def flatten(data):
    """
    Flatten deeply nested tuples into a one-layer tuple of elements

    params:
        data: nested tuples

    returns:
        result: flattened tuple
    """
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
    if show_arrow and N <= 50: 
        for i in range(N-1):
            dx = a[i+1, 0] - a[i, 0]
            dy = a[i+1, 1] - a[i, 1]
            plt.arrow(a[i, 0], a[i, 1], dx, dy, ls=':', fc='k', length_includes_head=True, shape='full', width=0.0002, head_width=0.02)
    ax.set_xlim([0.0, 1.0])
    ax.axis('equal')
    plt.tight_layout()
elif d == 3:
    ax = fig.add_subplot(projection='3d')
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], c=t, cmap='jet')
else:
    exit()
plt.savefig(f'{d}D_{N}.png', dpi=300)


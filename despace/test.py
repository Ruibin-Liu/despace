import sys
import numpy as np

from spatial_sort import SortND

s = SortND(sort_type='Morton')
data = np.array([[1.0, 0.1, 1.5, -0.3, 0.0], [1.5, 0.2, 1.3, -0.1, 0.7]]).transpose()
# data = [[i, j] for i in range(8) for j in range(8)]
# data = np.random.rand(2000, 2)
print("Initial:")
print(data)
atad = s.sort(data)
print("Final:")
print(atad)
s.plot(show_plot=False, plot_arrow=True)
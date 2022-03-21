from spatial_sort import SortND
import numpy as np
coords_1d = np.array([1.0, 0.1, 1.5, -0.3, 0.0])
s = SortND()
s(coords_1d)
s.plot()

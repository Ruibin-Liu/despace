from spatial_sort import SortND
import numpy as np
s = SortND(np.random.rand(40,2))
s.sort()
s.plot(plot_arrow=True)

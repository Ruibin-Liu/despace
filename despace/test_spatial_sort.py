from spatial_sort import SortND
import numpy as np
s = SortND(np.random.rand(40, 3))
s.sort()
s.plot(plot_arrow=True)
s(np.random.rand(50, 2))
s.plot(show_plot=True)

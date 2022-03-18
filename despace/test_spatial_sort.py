from spatial_sort import SortND
import numpy as np
s = SortND(np.random.rand(40,3))
s.sort()
s.plot(show_plot=True, plot_arrow=True)

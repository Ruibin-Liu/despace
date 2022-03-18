#!/usr/bin/env python3

from spatial_sort import SortND
import numpy as np

# init and sort
s = SortND(np.random.rand(50, 2))
s.sort()
s.plot(plot_arrow=True)

# directly call
t = SortND()
t(np.random.rand(50, 3))
t.plot(show_plot=True)

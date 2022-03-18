#!/usr/bin/env python3

import sys
import numpy as np

sys.path.append("..")
from despace.spatial_sort import SortND

# init and sort
s = SortND(np.random.rand(50, 3))
s.sort()
s.plot()

# directly call
t = SortND()
t(np.random.rand(50, 2))
t.plot(show_plot=True, plot_arrow=True)

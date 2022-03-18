#!/usr/bin/env python3

import sys
import numpy as np

sys.path.append("..")
from despace.spatial_sort import SortND

length = 50
d = 2
if len(sys.argv) > 1:
    length = int(sys.argv[1])
if len(sys.argv) > 2:
    d = int(sys.argv[2])

s = SortND(np.random.rand(length, d))
s.sort()
if length > 50:
    plot_arrow = False  # won't see the arrows anyway
else:
    plot_arrow = True
s.plot(show_plot=True, plot_arrow=plot_arrow)


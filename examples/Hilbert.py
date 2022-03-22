#!/usr/bin/env python3
"""
Construct Hilbert space-filling curves
"""
import sys
import numpy as np

sys.path.append("..")
from despace.spatial_sort import SortND

s = SortND(sort_type='Hilbert')
data = [[i, j] for i in range(8) for j in range(8)]
s.sort(data)
s.plot(show_plot=False, plot_arrow=True)


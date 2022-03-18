"""
    `despace` is a tool to sort n dimensional data in a spatial decomposition way. Instead of using space filling curves, we use a simple sorting scheme.

    For the n=1 case, it's just a single sort.
    For the n>=2 dimensional cases, it's possible to index 'coordinates' so that proximal points are stored closely. 

    We do it in the following way:
        1) Sort the the n-D coordinates by the first dimension;
        2) Divide the coordinates into two;
        3) For each half, sort the coordinates by the second dimension;
        4) For each sorted half, divide the coordinates into two, and for each half in halves, sort by the next dimension, and so on;
        5) Repeat by circularly rotating the dimension indices until divided to individual elements.

    The sorted (re-indexed) N-D array can be useful in geological and N-body simulations like in molecular dynamics and astronomical physics.
"""

import numpy as np
from math import floor
import matplotlib.pyplot as plt
from pkg_resources import get_distribution

__version__ = get_distribution('despace').version


class SortND:
    def __init__(self, coords=None, start_dim=0, plot_data=False, **kwargs):
        """
        Init an SortND instance.

        params:
            coords: numpy.ndarray like data; default is None.
            start_dim: int, sort the n-D array starting from the `start_dim` dimension; default is 0, the 0th dimension.
            plot_data: bool, whether to plot the data; it's automatically set to False if number of dimensions is > 3.
            Other defined variables.

        returns:
            a SortND instance
        """
        if coords is not None:
            self.orig_coords = np.array(coords)
            self.length, self.d = np.shape(self.orig_coords)
        self.start_dim = start_dim
        self.plot_data = plot_data
        self.__dict__.update(kwargs)

    def _next_dim(self, current_dim):
        """
        Generate the next dimension from the current dimension and the number of dimensions.

        params:
            current_dim: int, the current dimension in an array-like object.

        returns:
            dim_next: int, the next dimension in a circular way.
        """
        return current_dim + 1 if current_dim + 1 < self.d else 0

    def _sort_divide(self, coords, dim):
        """
        Sort an array 'coords' by its dimension 'dim', and then divide it into two halves.

        params:
            coords: numpy.ndarray-like, array to sort and divide.
            dim: int, the dimension to sort.

        returns:
            coords_1: first half of the sorted coords.
            coords_2: second half of the sorted coords.
        """
        N, _ = np.shape(coords)
        if N == 1:
            return coords[0]
        if self.d == 1:
            return np.sort(coords)
        dim_next = self._next_dim(dim)
        coords = coords[coords[:, dim].argsort()]
        half_index = floor(N / 2)
        return self._sort_divide(coords[0:half_index, :], dim_next), self._sort_divide(coords[half_index:N, :], dim_next)

    def _flatten(self, data):
        """
        Flatten deeply nested tuples into a one-layer tuple of elements.

        params:
            data: nested tuples.

        returns:
            result: flattened tuple.
        """
        result = []
        for item in data:
            if isinstance(item, tuple):
                result.extend(self._flatten(item))
            else:
                result.append(item)
        return tuple(result)

    def sort(self, new_coords=None):
        """
        Sort the n dimensional data.

        params:
            new_coords: new numpy.ndarray-like data to sort.

        returns:
            sorted_coords: sorted coords as a numpy.array.
        """
        if new_coords is not None:
            self.orig_coords = np.array(new_coords)
            self.length, self.d = np.shape(self.orig_coords)
        coords = self._sort_divide(self.orig_coords, self.start_dim)
        self.sorted_coords = np.stack(self._flatten(coords))
        return self.sorted_coords

    def plot(self, show_plot=False, save_plot=True, plot_arrow=False, **kwargs):
        """
        Plot the sorted data by using gradient color corresponding to the data index change.

        params:
            plot_arrow: bool, whether to plot arrows that show indexing direction between two data points; default is False. 
            show_plot: bool, whether to show the plot; default is False.
            save_plot: bool, whether to save the plot; default is True.
            Other matplotlib key words to control plot parameters.
        """
        self.fig_size = kwargs.get('figsize', (3.5, 3.5))
        self.color_map = kwargs.get('cmap', 'jet')
        self.scatter_size = kwargs.get('s', 20)
        self.fig_dpi = kwargs.get('dpi', 200)
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.plot_arrow = plot_arrow
        self.__dict__.update(kwargs)
        data = self.sorted_coords
        fig = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi)

        if self.plot_arrow:
            line_style = kwargs.get('linestyle', '--')
            self.ls = kwargs.get('ls', line_style)
            face_color = kwargs.get('facecolor', 'k')
            self.fc = kwargs.get('fc', face_color)
            self.width = kwargs.get('width', 0.002)
            self.head_width = kwargs.get('head_width', 0.02)

        if self.d in [1, 2]:
            ax = fig.add_subplot()
            ax.axis('equal')
        elif self.d == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            print("Cannot plot data when number of dimensions d > 3.")
            return False

        if self.d == 1:
            ax.scatter(data, data, s=self.scatter_size,
                       c=np.arange(self.length), cmap=self.color_map)
            if self.plot_arrow:
                for i in range(self.length-1):
                    dx = data[i+1] - data[i]
                    dy = dx
                    ax.arrow(data, data, dx, dy, ls=self.ls, fc=self.fc, length_includes_head=True,
                             shape='full', width=self.width, head_width=self.head_width)
        elif self.d == 2:
            ax.scatter(data[:, 0], data[:, 1], s=self.scatter_size,
                       c=np.arange(self.length), cmap=self.color_map)
            if self.plot_arrow:
                for i in range(self.length-1):
                    dx = data[i+1, 0] - data[i, 0]
                    dy = data[i+1, 1] - data[i, 1]
                    ax.arrow(data[i, 0], data[i, 1], dx, dy, ls=self.ls, fc=self.fc,
                             length_includes_head=True, shape='full', width=self.width, head_width=self.head_width)
        elif self.d == 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                       c=np.arange(self.length), cmap=self.color_map)
            if self.plot_arrow:
                for i in range(self.length-1):
                    dx = data[i+1, 0] - data[i, 0]
                    dy = data[i+1, 1] - data[i, 1]
                    dz = data[i+1, 2] - data[i, 2]
                    ax.quiver(data[i, 0], data[i, 1], data[i, 2],
                              dx, dy, dz, ls=self.ls, fc=self.fc)

        if self.d != 3:  # The z axis is often cutout in 3D projection plot
            plt.tight_layout()

        if self.save_plot:
            self.fig_format = kwargs.get('fig_format', 'png')
            self.transparent = kwargs.get('transparent', True)
            self.file_name = kwargs.get(
                'file_name', f"{self.d}D_{self.length}.{self.fig_format}")
            plt.savefig(self.file_name, dpi=self.fig_dpi,
                        transparent=self.transparent)

        if self.show_plot:
            plt.show()
        plt.close()

    def __call__(self, coords):
        """
        If called, return the sorted data.

        params:
            coords: numpy.ndarray-like data to sort.

        returns:
            sorted_coords: sorted coords as a numpy.array.
        """
        return self.sort(new_coords=coords)

    def __str__(self) -> str:
        return f"{SortND(N=self.length, d=self.d, start_dim=self.start_dim)}"

    def __repr__(self) -> str:
        return self.__str__()

"""Despace: Spatial Decomposition Without Space Filling Curves

    `Despace` is a tool to sort n dimensional data in a spatial decomposition way.
    Instead of using space filling curves, we use a simple sorting scheme.

    For the n=1 case, it's just a single sort.
    For the n>=2 dimensional cases, it's possible to index 'coordinates' so that proximal points are stored closely.

    We do it in the following way:
        1) Sort the the n-D coordinates by the first dimension;
        2) Divide the coordinates into two;
        3) For each half, sort the coordinates by the second dimension;
        4) For each sorted half, divide the coordinates into two, and for each half in halves, sort by the next dimension, and so on;
        5) Repeat by circularly rotating the dimension indices until divided to individual elements.

    The sorted (re-indexed) N-D array can be useful in geological and N-body simulations like in molecular dynamics
    and astronomical physics.
"""  # noqa: E501

from math import floor

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


class SortND:
    def __init__(
        self,
        coords: np.ndarray = None,
        start_dim: int = 0,
        sort_type: str = "Morton",
        plot_data: bool = False,
        **kwargs,
    ) -> None:
        """
        Init an SortND instance.

        params:
            coords: numpy.ndarray (2D) or matrix like data; default is None.
            start_dim: int, sort the matrix starting from the `start_dim` dimension; default is 0, the 0th dimension.
            sort_type: str, sorting type, either 'Morton' (Z-order) or 'Hilbert' (|_|-order).
            plot_data: bool, whether to plot the data; it's automatically set to False if number of dimensions is > 3.

        returns:
            a SortND instance
        """
        if coords is not None:
            self.coords = np.array(coords)
            shape = self.coords.shape
            if len(shape) == 1:
                self.length = shape[0]
                self.d = 1
            elif len(shape) == 2:
                self.length, self.d = shape[0], shape[1]
            else:
                raise ValueError(f"np.ndarray shape {shape} is not supported yet.")
        self.start_dim = start_dim
        self.sort_type = sort_type
        if self.sort_type not in ["Morton", "Hilbert"]:
            raise ValueError(
                f"Sort type {self.sort_type} is not supported. Only one of ['Morton', 'Hilbert'] is allowd."
            )
        self.plot_data = plot_data
        self.__dict__.update(kwargs)

    def _next_dim(self, current_dim: int) -> int:
        """
        Generate the next dimension from the current dimension and the number of dimensions.

        params:
            current_dim: int, the current dimension in an array-like object.

        returns:
            next_dim: int, the next dimension in a circular way.
        """
        return current_dim + 1 if current_dim + 1 < self.d else 0

    def _sort_divide_morton(
        self, coords: np.ndarray, dim: int, order: int
    ) -> np.ndarray:
        """
        Sort an array 'coords' by its dimension 'dim', and then divide it into two halves with the Morton-type order.

        params:
            coords: numpy.ndarray-like, array to sort and divide.
            dim: int, the dimension to sort.
            order: int, the sort oder for the current dimension

        returns:
            sorted_coords: sorted array according to spatial decomposition.
        """
        if self.d == 1:
            return np.sort(coords)
        shape = coords.shape
        N = shape[0]
        if N == 1:
            return coords[0]
        next_dim = self._next_dim(dim)
        coords = coords[coords[:, dim].argsort()]
        half_index = floor(N / 2)
        return np.vstack(
            (
                self._sort_divide_morton(coords[0:half_index, :], next_dim, order),
                self._sort_divide_morton(coords[half_index:N, :], next_dim, order),
            )
        )

    def _sort_divide_hilbert(
        self, coords: np.ndarray, sub_type: str = "A"
    ) -> np.ndarray:
        """
        Sort an array 'coords' by its dimension 'dim', and then divide it into two halves with the Hilbert-type order.

        params:
            coords: numpy.ndarray-like, array to sort and divide.
            sub_type: str, whether the sub structure is a 'A', 'B', 'C', or 'D' type;
                see https://en.wikipedia.org/wiki/Hilbert_curve#/media/File:Hilbert_curve_production_rules!.svg

        returns:
            sorted_coords: sorted array according to spatial decomposition.
        """
        if self.d == 1:
            return np.sort(coords)
        if self.d == 3:
            raise NotImplementedError("3D Hilbert type curve has not been implemented.")
        shape = coords.shape
        N = shape[0]
        if N == 1:
            return coords
        half_index = floor(N / 2)
        if sub_type == "A":
            coords = coords[coords[:, 0].argsort()]
            if coords.shape[0] == 2:
                return coords
            left, right = coords[0:half_index, :], coords[half_index:N, :]
            left = left[left[:, 1].argsort()]
            right = right[right[:, 1].argsort()[::-1]]
            half_left = floor(left.shape[0] / 2)
            half_right = floor(right.shape[0] / 2)
            down_left = left[0:half_left, :]
            up_left = left[half_left:half_index, :]
            up_right = right[0:half_right, :]
            down_right = right[half_right : (N - half_index), :]  # noqa: E203
            if left.shape[0] == 1:  # right length must be >= 2
                return np.vstack(
                    (
                        left,
                        self._sort_divide_hilbert(up_right, sub_type="A"),
                        self._sort_divide_hilbert(down_right, sub_type="B"),
                    )
                )
            elif (
                right.shape[0] == 1
            ):  # left length mush be >=2; same for other sub types
                return np.vstack(
                    (
                        self._sort_divide_hilbert(down_left, sub_type="D"),
                        self._sort_divide_hilbert(up_left, sub_type="A"),
                        right,
                    )
                )
            return np.vstack(
                (
                    self._sort_divide_hilbert(down_left, sub_type="D"),
                    self._sort_divide_hilbert(up_left, sub_type="A"),
                    self._sort_divide_hilbert(up_right, sub_type="A"),
                    self._sort_divide_hilbert(down_right, sub_type="B"),
                )
            )
        elif sub_type == "B":
            coords = coords[coords[:, 1].argsort()[::-1]]
            if coords.shape[0] == 2:
                return coords
            up, down = coords[0:half_index, :], coords[half_index:N, :]
            up = up[up[:, 0].argsort()[::-1]]
            down = down[down[:, 0].argsort()]
            half_up = floor(up.shape[0] / 2)
            half_down = floor(down.shape[0] / 2)
            up_right = up[0:half_up, :]
            up_left = up[half_up:half_index, :]
            down_left = down[0:half_down, :]
            down_right = down[half_down : (N - half_index), :]  # noqa: E203
            if up.shape[0] == 1:
                return np.vstack(
                    (
                        up,
                        self._sort_divide_hilbert(down_left, sub_type="B"),
                        self._sort_divide_hilbert(down_right, sub_type="A"),
                    )
                )
            elif down.shape[0] == 1:
                return np.vstack(
                    (
                        self._sort_divide_hilbert(up_right, sub_type="C"),
                        self._sort_divide_hilbert(up_left, sub_type="B"),
                        down,
                    )
                )
            return np.vstack(
                (
                    self._sort_divide_hilbert(up_right, sub_type="C"),
                    self._sort_divide_hilbert(up_left, sub_type="B"),
                    self._sort_divide_hilbert(down_left, sub_type="B"),
                    self._sort_divide_hilbert(down_right, sub_type="A"),
                )
            )
        elif sub_type == "C":
            coords = coords[coords[:, 0].argsort()[::-1]]
            if coords.shape[0] == 2:
                return coords
            right, left = coords[0:half_index, :], coords[half_index:N, :]
            right = right[right[:, 1].argsort()[::-1]]
            left = left[left[:, 1].argsort()]
            half_left = floor(left.shape[0] / 2)
            half_right = floor(right.shape[0] / 2)
            up_right = right[0:half_right, :]
            down_right = right[half_right:half_index, :]
            up_left = left[half_left : (N - half_index), :]  # noqa: E203
            down_left = left[0:half_left, :]
            if right.shape[0] == 1:
                return np.vstack(
                    (
                        right,
                        self._sort_divide_hilbert(down_left, sub_type="C"),
                        self._sort_divide_hilbert(up_left, sub_type="D"),
                    )
                )
            elif left.shape[0] == 1:
                return np.vstack(
                    (
                        self._sort_divide_hilbert(up_right, sub_type="B"),
                        self._sort_divide_hilbert(down_right, sub_type="C"),
                        left,
                    )
                )
            return np.vstack(
                (
                    self._sort_divide_hilbert(up_right, sub_type="B"),
                    self._sort_divide_hilbert(down_right, sub_type="C"),
                    self._sort_divide_hilbert(down_left, sub_type="C"),
                    self._sort_divide_hilbert(up_left, sub_type="D"),
                )
            )
        else:  # sub_type == 'D'
            coords = coords[coords[:, 1].argsort()]
            if coords.shape[0] == 2:
                return coords
            down, up = coords[0:half_index, :], coords[half_index:N, :]
            down = down[down[:, 0].argsort()]
            up = up[up[:, 0].argsort()[::-1]]
            half_up = floor(up.shape[0] / 2)
            half_down = floor(down.shape[0] / 2)
            down_left = down[0:half_down, :]
            down_right = down[half_down:half_index, :]
            up_right = up[0:half_up, :]
            up_left = up[half_up : (N - half_index), :]  # noqa: E203
            if down.shape[0] == 1:
                return np.vstack(
                    (
                        down,
                        self._sort_divide_hilbert(up_right, sub_type="D"),
                        self._sort_divide_hilbert(up_left, sub_type="C"),
                    )
                )
            elif up.shape[0] == 1:
                return np.vstack(
                    (
                        self._sort_divide_hilbert(down_left, sub_type="A"),
                        self._sort_divide_hilbert(down_right, sub_type="D"),
                        up,
                    )
                )
            return np.vstack(
                (
                    self._sort_divide_hilbert(down_left, sub_type="A"),
                    self._sort_divide_hilbert(down_right, sub_type="D"),
                    self._sort_divide_hilbert(up_right, sub_type="D"),
                    self._sort_divide_hilbert(up_left, sub_type="C"),
                )
            )

    def sort(self, coords: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Sort the n dimensional data.

        params:
            coords: numpy.ndarray-like data to sort.

        returns:
            sorted_coords: sorted coords as a numpy.array.
        """
        self.__dict__.update(kwargs)
        if coords is not None:
            self.coords = np.array(coords)
            shape = self.coords.shape
            if len(shape) == 1:
                self.length = shape[0]
                self.d = 1
            elif len(shape) == 2:
                self.length, self.d = shape[0], shape[1]
            else:
                raise ValueError(f"np.ndarray shape {shape} is not supported yet.")
        if self.sort_type == "Morton":
            self.sorted_coords = self._sort_divide_morton(
                self.coords, self.start_dim, 0
            )
        else:
            self.sorted_coords = self._sort_divide_hilbert(self.coords, "A")
        return self.sorted_coords

    def plot(
        self,
        show_plot: bool = False,
        save_plot: bool = True,
        plot_arrow: bool = False,
        **kwargs,
    ) -> bool:
        """
        Plot the sorted data by using gradient color corresponding to the data index change.

        params:
            plot_arrow: bool, whether to plot arrows that show indexing direction between
            two data points; default is False.
            show_plot: bool, whether to show the plot; default is False.
            save_plot: bool, whether to save the plot; default is True.
            Other matplotlib key words to control plot parameters.

        returns:
            True/False if the data is plotted or not.
        """
        self.fig_size = kwargs.get("figsize", (3.5, 3.5))
        self.color_map = kwargs.get("cmap", "jet")
        self.scatter_size = kwargs.get("s", 20)
        self.fig_dpi = kwargs.get("dpi", 200)
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.plot_arrow = plot_arrow
        self.__dict__.update(kwargs)
        data = self.sorted_coords
        fig = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi)

        if self.plot_arrow:
            line_style = kwargs.get("linestyle", "--")
            self.ls = kwargs.get("ls", line_style)
            face_color = kwargs.get("facecolor", "k")
            self.fc = kwargs.get("fc", face_color)
            self.width = kwargs.get("width", 0.002)
            self.head_width = kwargs.get("head_width", 0.02)

        if self.d in [1, 2]:
            ax = fig.add_subplot()
            ax.axis("equal")
        elif self.d == 3:
            ax = fig.add_subplot(projection="3d")
        else:
            raise ValueError(f"Cannot plot {self.d} dimensional data.")

        if self.d == 1:
            ax.scatter(
                data,
                data,
                s=self.scatter_size,
                c=np.arange(self.length),
                cmap=self.color_map,
            )
            if self.plot_arrow:
                for i in range(self.length - 1):
                    dx = data[i + 1] - data[i]
                    dy = dx
                    ax.arrow(
                        data,
                        data,
                        dx,
                        dy,
                        ls=self.ls,
                        fc=self.fc,
                        length_includes_head=True,
                        shape="full",
                        width=self.width,
                        head_width=self.head_width,
                    )
        elif self.d == 2:
            ax.scatter(
                data[:, 0],
                data[:, 1],
                s=self.scatter_size,
                c=np.arange(self.length),
                cmap=self.color_map,
            )
            if self.plot_arrow:
                for i in range(self.length - 1):
                    dx = data[i + 1, 0] - data[i, 0]
                    dy = data[i + 1, 1] - data[i, 1]
                    ax.arrow(
                        data[i, 0],
                        data[i, 1],
                        dx,
                        dy,
                        ls=self.ls,
                        fc=self.fc,
                        length_includes_head=True,
                        shape="full",
                        width=self.width,
                        head_width=self.head_width,
                    )
        elif self.d == 3:
            ax.scatter(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                c=np.arange(self.length),
                cmap=self.color_map,
            )
            if self.plot_arrow:
                for i in range(self.length - 1):
                    dx = data[i + 1, 0] - data[i, 0]
                    dy = data[i + 1, 1] - data[i, 1]
                    dz = data[i + 1, 2] - data[i, 2]
                    ax.quiver(
                        data[i, 0],
                        data[i, 1],
                        data[i, 2],
                        dx,
                        dy,
                        dz,
                        ls=self.ls,
                        fc=self.fc,
                    )

        if self.d != 3:  # The z axis is often cutout in 3D projection plot
            plt.tight_layout()

        if self.save_plot:
            self.fig_format = kwargs.get("fig_format", "png")
            self.transparent = kwargs.get("transparent", True)
            self.file_name = kwargs.get(
                "file_name", f"{self.d}D_{self.length}.{self.fig_format}"
            )
            plt.savefig(self.file_name, dpi=self.fig_dpi, transparent=self.transparent)

        if self.show_plot:
            plt.show()
        plt.close()
        return True

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        """
        If called, return the sorted data.

        params:
            coords: numpy.ndarray-like data to sort.

        returns:
            sorted_coords: sorted coords as a numpy.array.
        """
        self.__dict__.update(kwargs)
        return self.sort(coords=np.array(coords))

    def __str__(self) -> str:
        return f"SortND(N={self.length}, d={self.d}, start_dim={self.start_dim})"

    def __repr__(self) -> str:
        return self.__str__()

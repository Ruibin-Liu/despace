import sys
from os.path import exists
from unittest.mock import patch

import numpy as np  # type: ignore
import pytest

from despace.spatial_sort import SortND

sys.path.append("..")

coords_1d = np.array([1.0, 0.1, 1.5, -0.3, 0.0])
sorted_coords_1d = np.array([-0.3, 0.0, 0.1, 1.0, 1.5])

coords_2d = np.array(
    [[1.0, 0.1, 1.5, -0.3, 0.0], [1.5, 0.2, 1.3, -0.1, 0.7]]
).transpose()
sorted_coords_2d = np.array(
    [[-0.3, -0.1], [0.0, 0.7], [0.1, 0.2], [1.0, 1.5], [1.5, 1.3]]
)

coords_3d = np.array(
    [[1.2, 0.0, 1.7, -0.4, 0.1], [1.4, 0.9, 1.0, -0.6, 0.3], [2.0, 0.0, 1.4, -0.2, 0.2]]
).transpose()
sorted_coords_3d = np.array(
    [
        [-0.4, -0.6, -0.2],
        [0.0, 0.9, 0.0],
        [0.1, 0.3, 0.2],
        [1.7, 1.0, 1.4],
        [1.2, 1.4, 2.0],
    ]
)

grid_16 = np.array([[i, j] for i in range(4) for j in range(4)])
morton_grid_16 = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1],
        [2, 2],
        [2, 3],
        [3, 2],
        [3, 3],
    ]
)
hilbert_grid_16 = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 3],
        [1, 2],
        [2, 2],
        [2, 3],
        [3, 3],
        [3, 2],
        [3, 1],
        [2, 1],
        [2, 0],
        [3, 0],
    ]
)


def test_sort():
    # Init and call the sort method
    t = SortND(coords_1d)
    assert np.array_equal(t.sort(), sorted_coords_1d)
    t = SortND(coords_2d)
    assert np.array_equal(t.sort(), sorted_coords_2d)
    t = SortND(coords_3d)
    assert np.array_equal(t.sort(), sorted_coords_3d)
    with pytest.raises(ValueError):
        SortND(np.random.rand(2, 2, 2))

    # init and directly call
    s = SortND()
    assert np.array_equal(s(coords_1d), sorted_coords_1d)
    assert np.array_equal(s(coords_2d), sorted_coords_2d)
    assert np.array_equal(s(coords_3d), sorted_coords_3d)
    with pytest.raises(ValueError):
        s(np.random.rand(2, 2, 2))

    # test Morton
    s = SortND(sort_type="Morton")
    assert np.array_equal(s(grid_16), morton_grid_16)

    # test Hilbert
    s = SortND(sort_type="Hilbert")
    assert np.array_equal(s(grid_16), hilbert_grid_16)
    with pytest.raises(NotImplementedError):
        s(np.random.rand(5, 3))


@patch("matplotlib.pyplot.show")
def test_plot(mock_show):
    s = SortND()

    # show plots
    s(coords_1d)
    assert s.plot(save_plot=False)
    assert s.plot(save_plot=False, show_plot=True)
    s(coords_2d)
    assert s.plot(save_plot=False)
    assert s.plot(save_plot=False, show_plot=True)
    s(coords_3d)
    assert s.plot(save_plot=False)
    assert s.plot(save_plot=False, show_plot=True)

    # save plots
    s(coords_1d)
    s.plot(save_plot=True)
    assert exists("1D_5.png")
    s.plot(save_plot=True, file_name="test_1d.png")
    assert exists("test_1d.png")
    s(coords_2d)
    s.plot(save_plot=True)
    assert exists("2D_5.png")
    s.plot(save_plot=True, file_name="test_2d.png")
    assert exists("test_2d.png")
    s(coords_3d)
    s.plot(save_plot=True)
    assert exists("3D_5.png")
    s.plot(save_plot=True, file_name="test_3d.png")
    assert exists("test_3d.png")

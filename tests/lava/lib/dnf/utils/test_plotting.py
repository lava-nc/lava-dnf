# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from lava.lib.dnf.utils.plotting import raster_plot, \
    _compute_spike_rates, _compute_colored_spike_coordinates


class TestRasterPlot(unittest.TestCase):
    def test_compute_spike_rates(self) -> None:
        """Tests whether the instantaneous spike rates are computed as
        expected by the function used within raster_plot()."""
        spike_data = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        expected_spike_rates = [[0.25, 0.5, 0.0], [0.5, 0.5, 0.0],
                                [0.75, 0.25, 0.0], [0.5, 0.25, 0.0],
                                [0.5, 0.5, 0.0], [0.25, 0.25, 0.0],
                                [0.25, 0.25, 0.0], [0.5, 0.25, 0.0],
                                [0.5, 0.0, 0.0]]

        spike_rates = _compute_spike_rates(spike_data=np.array(spike_data),
                                           window_size=4)

        self.assertEqual(expected_spike_rates, spike_rates.tolist())

    def test_compute_colored_spike_coordinates(self) -> None:
        """Tests whether color information and coordinates are computed as
        expected by the function used within raster_plot()."""
        spike_data = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        spike_rates = [[0.25, 0.5, 0.0], [0.5, 0.5, 0.0],
                       [0.75, 0.25, 0.0], [0.5, 0.25, 0.0],
                       [0.5, 0.5, 0.0], [0.25, 0.25, 0.0],
                       [0.25, 0.25, 0.0], [0.5, 0.25, 0.0],
                       [0.5, 0.0, 0.0]]

        expected_x = [1, 2, 4, 4, 5, 7, 9]

        expected_y = [1, 0, 0, 1, 0, 1, 0]

        expected_colors = [0.5, 0.5, 0.5, 0.25, 0.5, 0.25, 0.5]

        x, y, colors = \
            _compute_colored_spike_coordinates(spike_data=np.array(spike_data),
                                               spike_rates=np.array(
                                                   spike_rates))

        self.assertEqual(expected_x, x)
        self.assertEqual(expected_y, y)
        self.assertEqual(expected_colors, colors)

    @patch("matplotlib.pyplot.show")
    def test_raster_plot_with_default_args(self,
                                           mock_show: MagicMock) -> None:
        """Tests whether the raster_plot function can be called with only
        spike_data as argument."""
        mock_show.return_value = None

        raster_plot(spike_data=np.zeros((10, 20)))

    @patch("matplotlib.pyplot.show")
    def test_raster_plot_with_non_default_args(self,
                                               mock_show: MagicMock) -> None:
        """Tests whether the raster_plot function can be called by also
        specifying the rate_window argument."""
        mock_show.return_value = None

        raster_plot(spike_data=np.zeros((10, 20)),
                    window_size=20)


if __name__ == '__main__':
    unittest.main()

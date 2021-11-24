# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from lava.lib.dnf.utils.plotting import raster_plot


class TestRasterPlot(unittest.TestCase):
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
        specifying the color and rate_window arguments."""
        mock_show.return_value = None

        raster_plot(spike_data=np.zeros((10, 20)),
                    color="#1f77b4",
                    rate_window=20)


if __name__ == '__main__':
    unittest.main()

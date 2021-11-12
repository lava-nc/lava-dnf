# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.dnf.kernels.kernels import Kernel


class TestKernel(unittest.TestCase):
    def setUp(self) -> None:
        """Instantiates a Kernel object."""
        self.kernel = Kernel(weights=np.zeros(1, ))

    def test_init(self):
        """Tests whether a Kernel can be instantiated."""
        self.assertIsInstance(self.kernel, Kernel)

    def test_weights_property(self):
        """Tests whether the weights can be accessed via a public property
        method."""
        self.assertTrue(np.array_equal(self.kernel.weights, np.zeros((1,))))

    def test_padding_value_property_and_default_value(self):
        """Tests whether the padding value can be accessed via a public
        property method."""
        self.assertEqual(self.kernel.padding_value, 0)

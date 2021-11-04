# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.dnf.utils.convenience import num_neurons, num_dims


class TestNumNeurons(unittest.TestCase):
    def test_num_neurons_1d(self):
        """Tests whether the number of neurons is computed correctly for
        one-dimensional shapes."""
        num = num_neurons(shape=(15,))
        self.assertEqual(num, 15)

    def test_num_neurons_2d(self):
        """Tests whether the number of neurons is computed correctly for
        two-dimensional shapes."""
        num = num_neurons(shape=(5, 3))
        self.assertEqual(num, 15)


class TestNumDims(unittest.TestCase):
    def test_num_dims_0d(self):
        """Tests whether dimensionality is computed correctly for
        zero-dimensional shapes."""
        dims = num_dims(shape=(1,))
        self.assertEqual(dims, 0)

    def test_num_neurons_1d(self):
        """Tests whether dimensionality is computed correctly for
        one-dimensional shapes."""
        dims = num_dims(shape=(10,))
        self.assertEqual(dims, 1)

    def test_num_neurons_2d(self):
        """Tests whether dimensionality is computed correctly for
        two-dimensional shapes."""
        dims = num_dims(shape=(10, 10))
        self.assertEqual(dims, 2)

if __name__ == '__main__':
    unittest.main()

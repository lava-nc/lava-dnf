# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.dnf.utils.convenience import num_neurons, num_dims, to_ndarray


class TestNumNeurons(unittest.TestCase):
    def test_num_neurons_1d(self) -> None:
        """Tests whether the number of neurons is computed correctly for
        one-dimensional shapes."""
        num = num_neurons(shape=(15,))
        self.assertEqual(num, 15)

    def test_num_neurons_2d(self) -> None:
        """Tests whether the number of neurons is computed correctly for
        two-dimensional shapes."""
        num = num_neurons(shape=(5, 3))
        self.assertEqual(num, 15)


class TestNumDims(unittest.TestCase):
    def test_num_dims_0d(self) -> None:
        """Tests whether dimensionality is computed correctly for
        zero-dimensional shapes."""
        dims = num_dims(shape=(1,))
        self.assertEqual(dims, 0)

    def test_num_neurons_1d(self) -> None:
        """Tests whether dimensionality is computed correctly for
        one-dimensional shapes."""
        dims = num_dims(shape=(10,))
        self.assertEqual(dims, 1)

    def test_num_neurons_2d(self) -> None:
        """Tests whether dimensionality is computed correctly for
        two-dimensional shapes."""
        dims = num_dims(shape=(10, 10))
        self.assertEqual(dims, 2)


class TestToNdarray(unittest.TestCase):
    def test_converting_float(self) -> None:
        """Tests whether floats can be converted to an ndarray."""
        a_float = 5.0
        ndarray = to_ndarray(a_float)
        self.assertIsInstance(ndarray, np.ndarray)
        self.assertTrue(np.array_equal(np.array([a_float]), ndarray))

    def test_converting_list(self) -> None:
        """Tests whether a list can be converted to an ndarray."""
        a_list = [1, 2, 3]
        ndarray = to_ndarray(a_list)
        self.assertIsInstance(ndarray, np.ndarray)
        self.assertTrue(np.array_equal(np.array(a_list), ndarray))

    def test_converting_tuple(self) -> None:
        """Tests whether a tuple can be converted to an ndarray."""
        a_tuple = (1, 2, 3)
        ndarray = to_ndarray(a_tuple)
        self.assertIsInstance(ndarray, np.ndarray)
        self.assertTrue(np.array_equal(np.array(a_tuple), ndarray))

    def test_ndarray_not_converted(self) -> None:
        """Tests whether an ndarray is simply returned."""
        ndarray = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(ndarray, to_ndarray(ndarray)))


if __name__ == '__main__':
    unittest.main()

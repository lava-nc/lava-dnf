# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lib.dnf.utils.convenience import num_neurons


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


if __name__ == '__main__':
    unittest.main()

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.dnf.utils.math import is_odd, gauss


class TestGauss(unittest.TestCase):
    def test_shape(self) -> None:
        """Tests whether the returned Gaussian has the specified shape."""
        shape = (5, 3)
        gaussian = gauss(shape)
        self.assertEqual(gaussian.shape, shape)

    def test_default_values(self) -> None:
        """Tests whether the default values for domain, amplitude, mean, and
        stddev are used."""
        shape = (3, 3)
        gaussian = gauss(shape)

        # amplitude should be 1, domain should be such that
        # the maximum is at position (0, 0), and mean should be 0
        self.assertEqual(gaussian[0, 0], 1)
        # stddev should be symmetrical
        self.assertEqual(gaussian[0, 1], gaussian[1, 0])
        self.assertEqual(gaussian[0, 2], gaussian[2, 0])
        self.assertEqual(gaussian[1, 2], gaussian[2, 1])

    def test_setting_amplitude(self) -> None:
        """Tests whether the amplitude is set to the correct value."""
        shape = (3, 3)
        amplitude = 42
        gaussian = gauss(shape, amplitude=amplitude)
        self.assertEqual(gaussian[0, 0], amplitude)

    def test_setting_domain(self) -> None:
        """Tests whether the domain is set correctly."""
        shape = (5,)
        domain = np.array([[-2.5, 2.5]])
        gaussian = gauss(shape, domain=domain)
        # with the specified domain, the maximum (1) should be in the center
        self.assertEqual(gaussian[2], 1)
        # and the gaussian should be symmetrical
        self.assertTrue(np.array_equal(gaussian, np.flip(gaussian)))

    def test_setting_mean(self) -> None:
        """Tests whether the mean is set correctly."""
        shape = (5,)
        mean = 1
        gaussian = gauss(shape, mean=mean)
        self.assertEqual(gaussian[mean], 1)

    def test_setting_stddev(self) -> None:
        """Tests whether the stddev can be set."""
        shape = (3,)
        gaussian_narrow = gauss(shape, stddev=1)
        gaussian_broad = gauss(shape, stddev=2)
        self.assertTrue(gaussian_narrow[1] < gaussian_broad[1])

    def test_domain_shape_mismatch_raises_error(self) -> None:
        """Tests whether an error is raised when the shape of the <domain>
        argument is different from <shape>."""
        shape = (5, 3)
        domain = np.zeros((1, 2))  # should be of shape (len(shape), 2)
        with self.assertRaises(ValueError):
            gauss(shape, domain=domain)

    def test_mean_shape_mismatch_raises_error(self) -> None:
        """Tests whether an error is raised when the shape of the <mean>
        argument is different from <shape>."""
        shape = (5, 3)
        mean = np.zeros((3,))  # should be of shape (2,)
        with self.assertRaises(ValueError):
            gauss(shape, mean=mean)

    def test_stddev_shape_mismatch_raises_error(self) -> None:
        """Tests whether an error is raised when the shape of the <stddev>
        argument is different from <shape>."""
        shape = (5, 3)
        stddev = np.ones((3,))  # should be of shape (2,)
        with self.assertRaises(ValueError):
            gauss(shape, stddev=stddev)


class TestIsOdd(unittest.TestCase):
    def test_is_odd(self) -> None:
        """Tests the is_odd() helper function."""
        self.assertFalse(is_odd(0))
        self.assertTrue(is_odd(1))


if __name__ == '__main__':
    unittest.main()

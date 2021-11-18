# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.dnf.kernels.kernels import Kernel, SelectiveKernel


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


class TestSelectiveKernel(unittest.TestCase):
    def test_init(self):
        """Tests whether a SelectiveKernel can be instantiated and arguments
        are set correctly."""
        amp_exc = 5.0
        width_exc = [2, 2]
        global_inh = -0.1
        kernel = SelectiveKernel(amp_exc=amp_exc,
                                 width_exc=width_exc,
                                 global_inh=global_inh)
        self.assertIsInstance(kernel, SelectiveKernel)
        self.assertEqual(kernel._amp_exc, amp_exc)
        self.assertEqual(kernel._global_inh, global_inh)
        self.assertTrue(np.array_equal(kernel._width_exc,
                                       np.array(width_exc)))
        self.assertEqual(kernel._limit, 1.0)

    def test_negative_amplitude_raises_error(self):
        """Tests whether a negative amplitude raises an error."""
        with self.assertRaises(ValueError):
            SelectiveKernel(amp_exc=-5.0, width_exc=[2, 2], global_inh=-0.1)

    def test_positive_global_inhibition_raises_error(self):
        """Tests whether a positive global inhibition raises an error."""
        with self.assertRaises(ValueError):
            SelectiveKernel(amp_exc=5.0, width_exc=[2, 2], global_inh=10.0)

    def test_negative_limit_raises_error(self):
        """Tests whether a negative limit raises an error."""
        with self.assertRaises(ValueError):
            SelectiveKernel(amp_exc=5.0, width_exc=[2, 2], global_inh=-0.1,
                            limit=-10)

    def test_shape_is_computed_from_width_and_limit(self):
        """Tests whether the shape of the kernel is computed correctly."""
        width = 4
        limit = 2
        kernel = SelectiveKernel(amp_exc=10,
                                 width_exc=width,
                                 global_inh=-5,
                                 limit=limit)

        self.assertEqual(kernel._shape[0], (2 * width * limit) + 1)

    def test_explicitly_specifying_odd_shape(self):
        """Tests whether specifying the shape of the kernel works."""
        shape = (5,)
        kernel = SelectiveKernel(amp_exc=10,
                                 width_exc=4,
                                 global_inh=-5,
                                 shape=shape)

        self.assertEqual(kernel._shape, shape)

    def test_explicitly_specified_shape_mismatching_width_raises_error(self):
        """Tests whether an error is raised when a shape is specified that
        is incompatible with the <width_exc> argument."""
        with self.assertRaises(ValueError):
            SelectiveKernel(amp_exc=10,
                            width_exc=[2, 2, 2],
                            global_inh=-5,
                            shape=(5, 3))

    def test_explicitly_specifying_even_shape_prints_warning(self):
        """Checks whether a warning is issued if the specified size of
        the kernel is even along one dimension."""
        shape = (4,)
        with self.assertWarns(Warning):
            kernel = SelectiveKernel(amp_exc=10,
                                     width_exc=4,
                                     global_inh=-5,
                                     shape=shape)

        self.assertTrue(kernel._shape, shape)

    def test_maximum_is_computed_correctly(self):
        """Checks whether the maximum of the kernel is computed correctly."""
        amp_exc = 10
        global_inh = -2
        size = 5
        kernel = SelectiveKernel(amp_exc=amp_exc,
                                 width_exc=4,
                                 global_inh=global_inh,
                                 shape=(size,))
        # weight at the center of the kernel should be amp_exc + global_inh
        center = int(size / 2)
        self.assertEqual(kernel.weights[center], amp_exc + global_inh)

    def test_computed_weights_when_dimensions_have_same_width_2d(self):
        """Checks whether the weight matrix has the same size in both
        dimensions if the width is specified to be equal."""
        kernel = SelectiveKernel(amp_exc=25,
                                 width_exc=[1.5, 1.5],
                                 global_inh=-1)

        self.assertEqual(kernel.weights.shape[0], kernel.weights.shape[1])

    def test_computed_weights_when_dimensions_have_different_width_2d(self):
        """Checks whether the weight matrix has a larger size when the width
        is specified as larger."""
        kernel = SelectiveKernel(amp_exc=25,
                                 width_exc=[4.0, 2.0],
                                 global_inh=-1)

        self.assertTrue(kernel.weights.shape[0] > kernel.weights.shape[1])

    def test_weight_symmetry(self):
        """Checks whether the computed kernel is symmetrical for different
        dimensionalities and widths."""
        widths_1d = [[wx] for wx in np.arange(1.0, 30.0, 3.0)]

        widths_2d = [[wx, wy]
                     for wx in np.arange(1.0, 30.0, 6.0)
                     for wy in np.arange(1.0, 40.0, 8.0)]

        widths_3d = [[wx, wy, wz]
                     for wx in np.arange(1.0, 30.0, 7.0)
                     for wy in np.arange(1.0, 40.0, 8.0)
                     for wz in np.arange(1.0, 50.0, 9.0)]

        for width in widths_1d + widths_2d + widths_3d:
            kernel = SelectiveKernel(amp_exc=25,
                                     width_exc=width,
                                     global_inh=-0.1)

            # kernel weights should always have an odd number of entries,
            # for every dimension
            for i in range(len(kernel.weights.shape)):
                self.assertTrue(np.size(kernel.weights, i) % 2)

            # kernel should be symmetric
            self.assertTrue(
                np.allclose(kernel.weights, np.flip(kernel.weights)))

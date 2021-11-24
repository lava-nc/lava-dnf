# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty

from lava.lib.dnf.kernels.kernels import Kernel, MultiPeakKernel, \
    SelectiveKernel, GaussianMixin


class TestKernel(unittest.TestCase):
    def setUp(self) -> None:
        """Instantiates a Kernel object."""
        self.kernel = Kernel(weights=np.zeros(1, ))

    def test_init(self) -> None:
        """Tests whether a Kernel can be instantiated."""
        self.assertIsInstance(self.kernel, Kernel)

    def test_weights_property(self) -> None:
        """Tests whether the weights can be accessed via a public property
        method."""
        self.assertTrue(np.array_equal(self.kernel.weights, np.zeros((1,))))

    def test_padding_value_property_and_default_value(self) -> None:
        """Tests whether the padding value can be accessed via a public
        property method."""
        self.assertEqual(self.kernel.padding_value, 0)


class TestGaussianMixin(unittest.TestCase):
    class MockKernel(GaussianMixin, Kernel):
        """Mock kernel to test the GaussianMixin"""
        def __init__(self,
                     amp_exc: float = 1.0,
                     width_exc: ty.Union[float, ty.List[float]] = 2.0,
                     limit: ty.Optional[float] = 1.0,
                     shape: ty.Optional[ty.Tuple[int, ...]] = None) -> None:
            GaussianMixin.__init__(self, amp_exc, width_exc, limit, shape)

        def _compute_weights(self) -> None:
            pass

    def test_negative_excitatory_amplitude_raises_error(self) -> None:
        """Tests whether a negative excitatory amplitude raises an error."""
        with self.assertRaises(ValueError):
            TestGaussianMixin.MockKernel(amp_exc=-5.0)

    def test_negative_limit_raises_error(self) -> None:
        """Tests whether a negative limit raises an error."""
        with self.assertRaises(ValueError):
            TestGaussianMixin.MockKernel(limit=-10)

    def test_computed_shape_is_always_odd(self) -> None:
        """Tests whether the computed shape always has an odd number of
        elements along each dimension."""
        for width in [2, 3, [2, 2], [3, 3], [2, 3]]:
            kernel = TestGaussianMixin.MockKernel(width_exc=width)
            self.assertFalse(np.any(np.array(kernel._shape) % 2 == 0))

    def test_explicitly_specifying_odd_shape(self) -> None:
        """Tests whether specifying the shape of the kernel works."""
        shape = (5,)
        kernel = TestGaussianMixin.MockKernel(shape=shape)
        self.assertEqual(kernel._shape, shape)

    def test_explicitly_specified_shape_mismatching_width_raises_error(self) \
            -> None:
        """Tests whether an error is raised when a shape is specified that
        is incompatible with the <width_exc> argument."""
        with self.assertRaises(ValueError):
            TestGaussianMixin.MockKernel(width_exc=[2, 2, 2],
                                         shape=(5, 3))

    def test_explicitly_specifying_even_shape_prints_warning(self) -> None:
        """Checks whether a warning is issued if the specified size of
        the kernel is even along one dimension."""
        shape = (4,)
        with self.assertWarns(Warning):
            kernel = TestGaussianMixin.MockKernel(shape=shape)
        self.assertTrue(kernel._shape, shape)


class TestMultiPeakKernel(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a MultiPeakKernel can be instantiated and arguments
        are set correctly."""
        amp_exc = 5.0
        width_exc = [2, 2]
        amp_inh = -2.0
        width_inh = [4, 4]
        kernel = MultiPeakKernel(amp_exc=amp_exc,
                                 width_exc=width_exc,
                                 amp_inh=amp_inh,
                                 width_inh=width_inh)
        self.assertIsInstance(kernel, MultiPeakKernel)
        self.assertEqual(kernel._amp_exc, amp_exc)
        self.assertTrue(np.array_equal(kernel._width_exc,
                                       np.array(width_exc)))
        self.assertEqual(kernel._amp_inh, amp_inh)
        self.assertTrue(np.array_equal(kernel._width_inh,
                                       np.array(width_inh)))
        self.assertEqual(kernel._limit, 1.0)
        self.assertEqual(kernel.padding_value, 0)

    def test_positive_inhibitory_amplitude_raises_error(self) -> None:
        """Tests whether a positive inhibitory amplitude raises an error."""
        with self.assertRaises(ValueError):
            MultiPeakKernel(amp_exc=5.0,
                            width_exc=[2, 2],
                            amp_inh=5.0,
                            width_inh=[4, 4])

    def test_widths_of_different_shape_raise_error(self) -> None:
        """Tests an error is raised when <width_exc> and <width_inh>
        have a different shape."""
        with self.assertRaises(ValueError):
            MultiPeakKernel(amp_exc=5.0,
                            width_exc=[2, 2],
                            amp_inh=-5.0,
                            width_inh=[4, 4, 4])

    def test_shape_is_computed_from_width_inh_and_limit(self) -> None:
        """Tests whether the shape of the kernel is computed correctly."""
        width_inh = 4
        limit = 2
        kernel = MultiPeakKernel(amp_exc=10,
                                 width_exc=2,
                                 amp_inh=-5,
                                 width_inh=width_inh,
                                 limit=limit)

        self.assertEqual(kernel._shape[0], (2 * width_inh * limit) + 1)

    def test_maximum_is_computed_correctly(self) -> None:
        """Checks whether the maximum of the kernel is computed correctly."""
        amp_exc = 10
        amp_inh = -2
        size = 5
        kernel = MultiPeakKernel(amp_exc=amp_exc,
                                 width_exc=4,
                                 amp_inh=amp_inh,
                                 width_inh=8,
                                 shape=(size,))
        # weight at the center of the kernel should be amp_exc + amp_inh
        center = int(size / 2)
        self.assertEqual(kernel.weights[center], amp_exc + amp_inh)

    def test_computed_weights_when_dimensions_have_same_width(self) -> None:
        """Checks whether the weight matrix has the same size in both
        dimensions if the inhibitory (!) width is specified to be equal."""
        kernel = MultiPeakKernel(amp_exc=25,
                                 width_exc=[1, 2],
                                 amp_inh=-10,
                                 width_inh=[3, 3])

        self.assertEqual(kernel.weights.shape[0], kernel.weights.shape[1])

    def test_computed_weights_when_dimensions_have_different_width_2d(self)\
            -> None:
        """Checks whether the weight matrix has a larger size when the
        inhibitory (!) width is specified as larger."""
        kernel = MultiPeakKernel(amp_exc=25,
                                 width_exc=[2, 2],
                                 amp_inh=-10,
                                 width_inh=[4, 2])

        self.assertTrue(kernel.weights.shape[0] > kernel.weights.shape[1])

    def test_weight_symmetry(self) -> None:
        """Checks whether the computed kernel is symmetrical for different
        dimensionalities and widths."""

        for width in [2, 3, [2, 3], [2, 3, 4]]:
            kernel = MultiPeakKernel(amp_exc=25,
                                     width_exc=width,
                                     amp_inh=-10,
                                     width_inh=width)

            self.assertTrue(np.allclose(kernel.weights,
                                        np.flip(kernel.weights)))


class TestSelectiveKernel(unittest.TestCase):
    def test_init(self) -> None:
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
        self.assertEqual(kernel.padding_value, global_inh)

    def test_positive_global_inhibition_raises_error(self) -> None:
        """Tests whether a positive global inhibition raises an error."""
        with self.assertRaises(ValueError):
            SelectiveKernel(amp_exc=5.0, width_exc=[2, 2], global_inh=10.0)

    def test_shape_is_computed_from_width_exc_and_limit(self) -> None:
        """Tests whether the shape of the kernel is computed correctly."""
        width = 4
        limit = 2
        kernel = SelectiveKernel(amp_exc=10,
                                 width_exc=width,
                                 global_inh=-5,
                                 limit=limit)

        self.assertEqual(kernel._shape[0], (2 * width * limit) + 1)

    def test_maximum_is_computed_correctly(self) -> None:
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

    def test_computed_weights_when_dimensions_have_same_width(self) -> None:
        """Checks whether the weight matrix has the same size in both
        dimensions if the width is specified to be equal."""
        kernel = SelectiveKernel(amp_exc=25,
                                 width_exc=[1.5, 1.5],
                                 global_inh=-1)

        self.assertEqual(kernel.weights.shape[0], kernel.weights.shape[1])

    def test_computed_weights_when_dimensions_have_different_width(self)\
            -> None:
        """Checks whether the weight matrix has a larger size when the width
        is specified as larger."""
        kernel = SelectiveKernel(amp_exc=25,
                                 width_exc=[4.0, 2.0],
                                 global_inh=-1)

        self.assertTrue(kernel.weights.shape[0] > kernel.weights.shape[1])

    def test_weight_symmetry(self) -> None:
        """Checks whether the computed kernel is symmetrical for different
        dimensionalities and widths."""

        for width in [2, 3, [2, 3], [2, 3, 4]]:
            kernel = SelectiveKernel(amp_exc=25,
                                     width_exc=width,
                                     global_inh=-0.1)

            self.assertTrue(np.allclose(kernel.weights,
                                        np.flip(kernel.weights)))

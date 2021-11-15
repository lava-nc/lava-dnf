# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty
import numpy as np

from lava.lib.dnf.inputs.inputs import GaussInputPattern, SpikeInputGenerator, BiasInputGenerator


class TestGaussInputPattern(unittest.TestCase):
    def test_init(self):
        """Tests whether GaussInputPattern can be instantiated"""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)

        self.assertIsInstance(input_pattern, GaussInputPattern)

    def test_missing_param_raises_type_error(self):
        """Tests whether a missing parameters at construction raises a Type error"""
        with self.assertRaises(TypeError):
            GaussInputPattern(amplitude=2.0, mean=30, stddev=3)

        with self.assertRaises(TypeError):
            GaussInputPattern(shape=(60,), mean=30, stddev=3)

        with self.assertRaises(TypeError):
            GaussInputPattern(shape=(60,), amplitude=2.0, stddev=3)

        with self.assertRaises(TypeError):
            GaussInputPattern(shape=(60,), amplitude=2.0, mean=30)

    def test_properties_accessible(self):
        """Tests whether properties are accessible"""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)

        raised = False

        try:
            _ = input_pattern.shape
            _ = input_pattern.pattern
            _ = input_pattern.amplitude
            _ = input_pattern.mean
            _ = input_pattern.stddev
        except AttributeError:
            raised = True

        self.assertFalse(raised)

    def test_mean_shape_mismatch_with_shape_dimensionality(self):
        """Tests whether instantiating a GaussInputPattern with an mean array of shape mismatching the shape
            parameter's dimensionality raises a Value Error"""
        with self.assertRaises(ValueError):
            GaussInputPattern(shape=(60, 60), amplitude=2.0, mean=[30, 30, 30], stddev=3)

    def test_stddev_shape_mismatch_with_shape_dimensionality(self):
        """Tests whether instantiating a GaussInputPattern with an stddev array of shape mismatching the shape
            parameter's dimensionality raises a Value Error"""
        with self.assertRaises(ValueError):
            GaussInputPattern(shape=(60, 60), amplitude=2.0, mean=30, stddev=[3, 3, 3])

    def test_float_mean_and_std_are_broadcast_when_multi_dim_shape(self):
        """Tests whether float number mean and stddev get broadcasted to ndarray with shape matching
            the dimensionality of the shape parameter"""
        input_pattern = GaussInputPattern(shape=(60, 60), amplitude=2.0, mean=30, stddev=3)

        self.assertEqual(input_pattern.mean, [30, 30])
        self.assertEqual(input_pattern.stddev, [3, 3])

    def test_parameters_are_set_correctly(self):
        """Tests whether float number mean and stddev get broadcasted to ndarray with shape matching
            the dimensionality of the shape parameter"""
        input_pattern = GaussInputPattern(shape=(60, 60), amplitude=2.0, mean=30, stddev=3)

        self.assertEqual(input_pattern.shape, (60, 60))
        self.assertEqual(input_pattern.amplitude, 2.0)
        self.assertEqual(input_pattern.mean, [30, 30])
        self.assertEqual(input_pattern.stddev, [3, 3])

    def test_pattern_shape_equal_to_shape(self):
        """Tests whether shape of generated pattern is equal to shape parameter"""
        input_pattern = GaussInputPattern(shape=(60, 60), amplitude=2.0, mean=30, stddev=3)

        self.assertEqual(input_pattern.pattern.shape, input_pattern.shape)


class TestSpikeInputGenerator(unittest.TestCase):
    def test_init(self):
        """Tests whether SpikeInputGenerator can be instantiated"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)

        self.assertIsInstance(spike_generator, SpikeInputGenerator)

    def test_shape_property(self):
        """Tests whether shape property is accessible and with the right value"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)

        self.assertEqual(spike_generator.shape, (60, ))

    def test_generate_return_type(self):
        """Tests whether generate method return type is ndarray"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)

        self.assertIsInstance(spike_generator.generate(time_step=0), np.ndarray)

    def test_generate_returned_shape(self):
        """Tests whether generate method return an ndarray with the correct shape"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)

        self.assertEqual(spike_generator.shape, spike_generator.generate(time_step=0).shape)


class TestBiasInputGenerator(unittest.TestCase):
    def test_init(self):
        """Tests whether BiasInputGenerator can be instantiated"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        bias_generator = BiasInputGenerator(input_pattern)

        self.assertIsInstance(bias_generator, BiasInputGenerator)

    def test_shape_property(self):
        """Tests whether shape property is accessible and with the right value"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        bias_generator = BiasInputGenerator(input_pattern)

        self.assertEqual(bias_generator.shape, (60, ))

    def test_generate_return_type(self):
        """Tests whether generate method return type is ndarray"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        bias_generator = BiasInputGenerator(input_pattern)

        self.assertIsInstance(bias_generator.generate(), np.ndarray)

    def test_generate_returned_shape(self):
        """Tests whether generate method return an ndarray with the correct shape"""
        input_pattern = GaussInputPattern(shape=(60, ), amplitude=2.0, mean=30, stddev=3)
        bias_generator = BiasInputGenerator(input_pattern)

        self.assertEqual(bias_generator.shape, bias_generator.generate().shape)


if __name__ == '__main__':
    unittest.main()

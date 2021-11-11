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
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)

        self.assertIsInstance(input_pattern, GaussInputPattern)

    def test_missing_param_raises_type_error(self):
        """Tests whether a missing parameters at construction raises a Type error"""
        with self.assertRaises(TypeError):
            GaussInputPattern(mean=30, stddev=3)

        with self.assertRaises(TypeError):
            GaussInputPattern(amplitude=2.0, stddev=3)

        with self.assertRaises(TypeError):
            GaussInputPattern(amplitude=2.0, mean=30)

        with self.assertRaises(TypeError):
            GaussInputPattern(amplitude=2.0)

        with self.assertRaises(TypeError):
            GaussInputPattern(mean=30)

        with self.assertRaises(TypeError):
            GaussInputPattern(stddev=3)

    def test_is_callable(self):
        """Tests whether GaussInputPattern is callable"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)

        self.assertTrue(callable(input_pattern))

    def test_call_without_shape_raises_type_error(self):
        """Tests whether calling the GaussInputPattern object without shape argument raises a Type error"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)

        with self.assertRaises(TypeError):
            input_pattern()

    def test_call_return_type(self):
        """Tests whether the return type of the call to the GaussInputPattern object is ndarray"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )

        self.assertIsInstance(input_pattern(shape), np.ndarray)

    def test_call_returned_shape(self):
        """Tests whether the return type of the call to the GaussInputPattern object is ndarray"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )

        self.assertEqual(shape, input_pattern(shape).shape)

    def test_params_can_be_accessed(self):
        """Tests whether parameters can be updated after instantiation"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)

        # Should we reqlly test for these ? (private variable)
        self.assertEqual(2.0, input_pattern._pattern_params["amplitude"])
        self.assertEqual(30, input_pattern._pattern_params["mean"])
        self.assertEqual(3, input_pattern._pattern_params["stddev"])

    def test_params_can_be_updated(self):
        """Tests whether parameters can be updated after instantiation"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)

        input_pattern.set_params(amplitude=1.0)
        self.assertEqual(1.0, input_pattern._pattern_params["amplitude"])

        input_pattern.set_params(mean=20)
        self.assertEqual(20, input_pattern._pattern_params["mean"])

        input_pattern.set_params(stddev=2)
        self.assertEqual(2, input_pattern._pattern_params["stddev"])

        input_pattern.set_params(amplitude=2.0, mean=30)
        self.assertEqual(2.0, input_pattern._pattern_params["amplitude"])
        self.assertEqual(30, input_pattern._pattern_params["mean"])

        input_pattern.set_params(amplitude=3.0, mean=40, stddev=5)
        self.assertEqual(3.0, input_pattern._pattern_params["amplitude"])
        self.assertEqual(40, input_pattern._pattern_params["mean"])
        self.assertEqual(5, input_pattern._pattern_params["stddev"])


class TestSpikeInputGenerator(unittest.TestCase):
    def test_init(self):
        """Tests whether SpikeInputGenerator can be instantiated"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )
        spike_generator = SpikeInputGenerator(shape, input_pattern)

        self.assertIsInstance(spike_generator, SpikeInputGenerator)

    def test_generate_return_type(self):
        """Tests whether generate method return type is ndarray"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )
        spike_generator = SpikeInputGenerator(shape, input_pattern)

        self.assertIsInstance(spike_generator.generate(time_step=0), np.ndarray)

    def test_generate_returned_shape(self):
        """Tests whether generate method return an ndarray with the correct shape"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )
        spike_generator = SpikeInputGenerator(shape, input_pattern)

        self.assertEqual(shape, spike_generator.generate(time_step=0).shape)

class TestBiasInputGenerator(unittest.TestCase):
    def test_init(self):
        """Tests whether BiasInputGenerator can be instantiated"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )
        bias_generator = BiasInputGenerator(shape, input_pattern)

        self.assertIsInstance(bias_generator, BiasInputGenerator)

    def test_generate_return_type(self):
        """Tests whether generate method return type is ndarray"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )
        bias_generator = BiasInputGenerator(shape, input_pattern)

        self.assertIsInstance(bias_generator.generate(), np.ndarray)

    def test_generate_returned_shape(self):
        """Tests whether generate method return an ndarray with the correct shape"""
        input_pattern = GaussInputPattern(amplitude=2.0, mean=30, stddev=3)
        shape = (60, )
        bias_generator = BiasInputGenerator(shape, input_pattern)

        self.assertEqual(shape, bias_generator.generate().shape)


if __name__ == '__main__':
    unittest.main()

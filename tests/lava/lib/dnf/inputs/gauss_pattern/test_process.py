# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern


class TestGaussPattern(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a GaussPattern process can be initiated."""
        gauss_pattern = GaussPattern(shape=(30, 30),
                                     amplitude=200.,
                                     mean=[15., 15.],
                                     stddev=[5., 5.])

        np.testing.assert_array_equal(gauss_pattern.shape,
                                      np.array((30, 30)))
        np.testing.assert_array_equal(gauss_pattern.amplitude,
                                      np.array([200.]))
        np.testing.assert_array_equal(gauss_pattern.mean,
                                      np.array([15., 15.]))
        np.testing.assert_array_equal(gauss_pattern.stddev,
                                      np.array([5., 5.]))
        np.testing.assert_array_equal(gauss_pattern.null_pattern.get(),
                                      np.full((30, 30), np.nan))
        np.testing.assert_array_equal(gauss_pattern.pattern.get(),
                                      np.zeros((30, 30)))
        np.testing.assert_array_equal(gauss_pattern.changed.get(),
                                      np.array([True]))

    def test_init_float_parameters(self) -> None:
        """Tests whether a GaussPattern process can be initiated with float
        mean and stddev."""
        gauss_pattern = GaussPattern(shape=(30, 30),
                                     amplitude=200.,
                                     mean=15.,
                                     stddev=5.)

        np.testing.assert_array_equal(gauss_pattern.shape,
                                      np.array((30, 30)))
        np.testing.assert_array_equal(gauss_pattern.amplitude,
                                      np.array([200.]))
        np.testing.assert_array_equal(gauss_pattern.mean,
                                      np.array([15., 15.]))
        np.testing.assert_array_equal(gauss_pattern.stddev,
                                      np.array([5., 5.]))
        np.testing.assert_array_equal(gauss_pattern.null_pattern.get(),
                                      np.full((30, 30), np.nan))
        np.testing.assert_array_equal(gauss_pattern.pattern.get(),
                                      np.zeros((30, 30)))
        np.testing.assert_array_equal(gauss_pattern.changed.get(),
                                      np.array([True]))

    def test_init_validation(self) -> None:
        """Tests whether a GaussPattern process instantiation with mean or
        stddev length not matching shape dimensionality raises a ValueError."""
        with self.assertRaises(ValueError):
            GaussPattern(shape=(30, 30),
                         amplitude=200.,
                         mean=[15., 15., 15.],
                         stddev=5.)

        with self.assertRaises(ValueError):
            GaussPattern(shape=(30, 30),
                         amplitude=200.,
                         mean=15.,
                         stddev=[5., 5., 5.])

    def test_running(self) -> None:
        """Tests whether a GaussPattern process can be run."""
        num_steps = 10

        gauss_pattern = GaussPattern(shape=(30, 30),
                                     amplitude=200.,
                                     mean=[15., 15.],
                                     stddev=[5., 5.])

        try:
            gauss_pattern.run(condition=RunSteps(num_steps=num_steps),
                              run_cfg=Loihi1SimCfg())
        finally:
            gauss_pattern.stop()

        self.assertEqual(gauss_pattern.runtime.num_steps, num_steps)

    def test_set_parameters(self) -> None:
        """Tests whether setters for amplitude, mean and stddev actually
        set values for the corresponding Vars and whether the changed Var
        gets set to True."""
        gauss_pattern = GaussPattern(shape=(30, 30),
                                     amplitude=200.,
                                     mean=[15., 15.],
                                     stddev=[5., 5.])

        try:
            gauss_pattern.run(condition=RunSteps(num_steps=1),
                              run_cfg=Loihi1SimCfg())

            gauss_pattern.amplitude = 250.

            np.testing.assert_array_equal(gauss_pattern.amplitude,
                                          np.array([250.]))
            np.testing.assert_array_equal(gauss_pattern.changed.get(),
                                          np.array([True]))

            gauss_pattern.run(condition=RunSteps(num_steps=1),
                              run_cfg=Loihi1SimCfg())

            gauss_pattern.mean = 20.

            np.testing.assert_array_equal(gauss_pattern.mean,
                                          np.array([20., 20.]))
            np.testing.assert_array_equal(gauss_pattern.changed.get(),
                                          np.array([True]))

            gauss_pattern.run(condition=RunSteps(num_steps=1),
                              run_cfg=Loihi1SimCfg())

            gauss_pattern.mean = [22., 22.]

            np.testing.assert_array_equal(gauss_pattern.mean,
                                          np.array([22., 22.]))
            np.testing.assert_array_equal(gauss_pattern.changed.get(),
                                          np.array([True]))

            gauss_pattern.run(condition=RunSteps(num_steps=1),
                              run_cfg=Loihi1SimCfg())

            gauss_pattern.stddev = 10.

            np.testing.assert_array_equal(gauss_pattern.stddev,
                                          np.array([10., 10.]))
            np.testing.assert_array_equal(gauss_pattern.changed.get(),
                                          np.array([True]))

            gauss_pattern.run(condition=RunSteps(num_steps=1),
                              run_cfg=Loihi1SimCfg())

            gauss_pattern.stddev = [13., 13.]

            np.testing.assert_array_equal(gauss_pattern.stddev,
                                          np.array([13., 13.]))
            np.testing.assert_array_equal(gauss_pattern.changed.get(),
                                          np.array([True]))
        finally:
            gauss_pattern.stop()

    def test_set_parameters_validation(self) -> None:
        """Tests whether setters for mean and stddev raise a ValueError for
        lengths not matching shape dimensionality."""
        gauss_pattern = GaussPattern(shape=(30, 30),
                                     amplitude=200.,
                                     mean=[15., 15.],
                                     stddev=[5., 5.])

        try:
            gauss_pattern.run(condition=RunSteps(num_steps=1),
                              run_cfg=Loihi1SimCfg())

            with self.assertRaises(ValueError):
                gauss_pattern.mean = [10., 10., 10.]

            with self.assertRaises(ValueError):
                gauss_pattern.stddev = [10., 10., 10.]
        finally:
            gauss_pattern.stop()


if __name__ == '__main__':
    unittest.main()

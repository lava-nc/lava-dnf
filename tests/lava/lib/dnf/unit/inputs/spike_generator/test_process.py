# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.dnf.inputs.spike_generator.process import SpikeGenerator


class TestSpikeGenerator(unittest.TestCase):
    def test_init(self):
        """Tests whether a SpikeGenerator process can be initiated."""
        spike_generator = SpikeGenerator(shape=(30, 30))

        np.testing.assert_array_equal(
            spike_generator.inter_spike_distances.get(), np.zeros((30, 30)))
        np.testing.assert_array_equal(
            spike_generator.first_spike_times.get(), np.zeros((30, 30)))
        np.testing.assert_array_equal(
            spike_generator.last_spiked.get(), np.full((30, 30), -np.inf))
        np.testing.assert_array_equal(
            spike_generator.spikes.get(), np.zeros((30, 30)))

    def test_init_validation(self):
        """Tests whether a SpikeGenerator process instantiation with
        non-valid min_spike_rate or seed raises a ValueError."""
        with self.assertRaises(ValueError):
            SpikeGenerator(shape=(30, 30), min_spike_rate=-5)

        with self.assertRaises(ValueError):
            SpikeGenerator(shape=(30, 30), min_spike_rate=-5)

    def test_running(self):
        """Tests whether a SpikeGenerator process can be run."""
        num_steps = 10

        spike_generator = SpikeGenerator(shape=(30, 30))

        try:
            spike_generator.run(condition=RunSteps(num_steps=num_steps),
                                run_cfg=Loihi1SimCfg())
        finally:
            spike_generator.stop()

        self.assertEqual(spike_generator.runtime.current_ts, num_steps)


if __name__ == '__main__':
    unittest.main()

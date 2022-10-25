# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen


class TestRateCodeSpikeGen(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a RateCodeSpikeGen process can be initiated."""
        spike_generator = RateCodeSpikeGen(shape=(30, 30))

        np.testing.assert_array_equal(
            spike_generator.inter_spike_distances.get(),
            np.zeros((30, 30))
        )
        np.testing.assert_array_equal(
            spike_generator.first_spike_times.get(),
            np.zeros((30, 30))
        )
        np.testing.assert_array_equal(
            spike_generator.last_spiked.get(),
            np.full((30, 30), -np.inf)
        )
        np.testing.assert_array_equal(
            spike_generator.spikes.get(),
            np.zeros((30, 30))
        )

    def test_negative_min_spike_rate_raises_error(self) -> None:
        """Tests whether specifying a negative min_spike_rate raises an
        error."""
        with self.assertRaises(ValueError):
            RateCodeSpikeGen(shape=(30, 30), min_spike_rate=-5)

    def test_negative_seed_raises_error(self) -> None:
        """Tests whether specifying a negative seed raises an error."""
        with self.assertRaises(ValueError):
            RateCodeSpikeGen(shape=(30, 30), seed=-5)


if __name__ == '__main__':
    unittest.main()

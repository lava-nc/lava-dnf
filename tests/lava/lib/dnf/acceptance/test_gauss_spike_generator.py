# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.sink import RingBuffer

from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen


class TestGaussRateCodeSpikeGen(unittest.TestCase):
    def test_rate_code_spike_gen_receiving_gauss_pattern(self) -> None:
        """Tests whether the SpikeGenerator Process works as expected in
        combination with the GaussPattern Process, producing spikes that are
        centered around neuron 1 for 10 time steps, and then switching to
        spikes centered around neuron 3 for subsequent 10 time steps."""
        num_steps_per_pattern = 10
        shape = (5,)
        expected_spikes = np.array(
            [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )

        gauss_pattern = GaussPattern(shape=shape, amplitude=1500.0, mean=1,
                                     stddev=0.2)
        spike_gen = RateCodeSpikeGen(shape=shape, seed=42)
        sink_process = RingBuffer(shape=(shape[0],),
                                  buffer=num_steps_per_pattern * 2)

        gauss_pattern.out_ports.a_out.connect(spike_gen.in_ports.a_in)
        spike_gen.out_ports.s_out.connect(sink_process.in_ports.a_in)

        run_condition = RunSteps(num_steps=num_steps_per_pattern)
        run_cfg = Loihi1SimCfg()

        try:
            spike_gen.run(condition=run_condition, run_cfg=run_cfg)
            gauss_pattern.mean = 3
            spike_gen.run(condition=run_condition, run_cfg=run_cfg)

            received_spikes = sink_process.data.get()
            np.testing.assert_array_equal(received_spikes,
                                          expected_spikes)
        finally:
            spike_gen.stop()


if __name__ == '__main__':
    unittest.main()

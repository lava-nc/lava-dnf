# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.dnf.inputs.spike_source.process import SpikeSource
from lava.lib.dnf.inputs.inputs import GaussInputPattern, SpikeInputGenerator


class TestSpikeSource(unittest.TestCase):
    def test_init(self):
        """Tests whether a SpikeSource process can be initiated."""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)
        spike_source = SpikeSource(generator=spike_generator)

        self.assertIsInstance(spike_source, SpikeSource)

    def test_init_without_generator(self):
        """Tests whether a SpikeSource instantiation without spike generator raises an ."""
        with self.assertRaises(KeyError):
            SpikeSource()

    def test_shape_is_set_correctly(self):
        """Tests whether shape is set by default."""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)
        spike_source = SpikeSource(generator=spike_generator)

        self.assertEqual(spike_source.shape, spike_generator.shape)

    def test_ports_exist(self):
        """Tests whether the OutPort is created."""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)
        spike_source = SpikeSource(generator=spike_generator)

        self.assertIsInstance(spike_source.out_ports.s_out, OutPort)
        self.assertEqual(spike_source.out_ports.s_out.shape,spike_source.shape)

    def test_running(self):
        """Tests whether a SpikeSource process can be executed."""
        num_steps = 10

        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)
        spike_source = SpikeSource(generator=spike_generator)

        spike_source.run(condition=RunSteps(num_steps=num_steps),
                         run_cfg=Loihi1SimCfg())
        spike_source.stop()

        self.assertEqual(spike_source.runtime.current_ts, num_steps)


if __name__ == '__main__':
    unittest.main()

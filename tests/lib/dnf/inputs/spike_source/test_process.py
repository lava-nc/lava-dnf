# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.dnf.inputs.spike_source.process import SpikeSource


class TestSpikeSource(unittest.TestCase):
    def test_init(self):
        """Tests whether a SpikeSource process can be initiated."""
        spike_source = SpikeSource()
        self.assertIsInstance(spike_source, SpikeSource)

    def test_default_shape(self):
        """Tests whether shape is set by default."""
        spike_source = SpikeSource()
        self.assertEqual(spike_source.shape, (1,))

    def test_setting_shape(self):
        """Tests whether setting the shape works."""
        shape = (5, 3)
        spike_source = SpikeSource(shape=shape)
        self.assertEqual(spike_source.shape, shape)

    def test_ports_exist(self):
        """Tests whether the OutPort is created."""
        shape = (5, 3)
        spike_source = SpikeSource(shape=shape)
        self.assertIsInstance(spike_source.out_ports.s_out, OutPort)
        self.assertEqual(spike_source.out_ports.s_out.shape, shape)

    # def test_running(self):
    #     """Tests whether a SpikeSource process can be executed."""
    #     num_steps = 10
    #     spike_source = SpikeSource(shape=(5, 3))
    #     spike_source.run(condition=RunSteps(num_steps=num_steps),
    #                      run_cfg=Loihi1SimCfg(select_sub_proc_model=True))
    #     spike_source.stop()

    #     self.assertEqual(spike_source.runtime.current_ts, num_steps)


if __name__ == '__main__':
    unittest.main()

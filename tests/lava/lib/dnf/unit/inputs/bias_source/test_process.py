# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest


from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.dnf.inputs.bias_source.process import BiasSource


class TestBiasSource(unittest.TestCase):
    def test_init(self):
        """Tests whether a BiasSource process can be initiated."""
        bias_source = BiasSource()
        self.assertIsInstance(bias_source, BiasSource)

    def test_default_shape(self):
        """Tests whether shape is set by default."""
        bias_source = BiasSource()
        self.assertEqual(bias_source.shape, (1,))

    def test_setting_shape(self):
        """Tests whether setting the shape works."""
        shape = (5, 3)
        bias_source = BiasSource(shape=shape)
        self.assertEqual(bias_source.shape, shape)

    def test_ports_exist(self):
        """Tests whether the RefPort is created."""
        shape = (5, 3)
        bias_source = BiasSource(shape=shape)
        self.assertIsInstance(bias_source.ref_ports.b_out, RefPort)
        self.assertEqual(bias_source.ref_ports.b_out.shape, shape)

    # def test_running(self):
    #     """Tests whether a BiasSource process can be executed."""
    #     num_steps = 10
    #     bias_source = BiasSource(shape=(5, 3))
    #     bias_source.run(condition=RunSteps(num_steps=num_steps),
    #                      run_cfg=Loihi1SimCfg(select_sub_proc_model=True))
    #     bias_source.stop()

    #     self.assertEqual(bias_source.runtime.current_ts, num_steps)


if __name__ == '__main__':
    unittest.main()

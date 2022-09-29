# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import sys

from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.lif.process import LIF
from lava.proc.embedded_io.spike import PyToNxAdapter
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from lava.lib.dnf.kernels.kernels import SelectiveKernel
from lava.lib.dnf.operations.operations import Convolution, Weights
from lava.lib.dnf.connect.connect import connect

from tests.lava.test_utils.utils import Utils

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
run_loihi_tests: bool = Utils.get_bool_env_setting("RUN_LOIHI_TESTS")


class TestDNFOnLoihi2(unittest.TestCase):
    @unittest.skipUnless(run_loihi_tests, 'RUN_LOIHI_TESTS is not enabled.')
    def test_1d_dnf_on_loihi2(self) -> None:
        num_steps = 10
        num_neurons = 10
        shape = (num_neurons,)

        gauss_pattern_params = {"shape": shape,
                                "amplitude": 3000,
                                "mean": 5,
                                "stddev": 3}
        spike_gen_params = {"shape": shape,
                            "seed": 1}
        dnf_params = {"shape": shape,
                      "du": 409,
                      "dv": 2047,
                      "vth": 50}
        kernel_params = {"amp_exc": 18,
                         "width_exc": 3,
                         "global_inh": -15}

        ### Python ###

        gauss_pattern = GaussPattern(**gauss_pattern_params)
        spike_generator = RateCodeSpikeGen(**spike_gen_params)
        dnf = LIF(**dnf_params)
        kernel = SelectiveKernel(**kernel_params)

        gauss_pattern.a_out.connect(spike_generator.a_in)
        connect(spike_generator.s_out, dnf.a_in, ops=[Weights(20)])
        connect(dnf.s_out, dnf.a_in, ops=[Convolution(kernel)])

        voltages_py = np.zeros((num_neurons, num_steps), dtype=int)
        try:
            for i in range(num_steps):
                dnf.run(condition=RunSteps(num_steps=1),
                        run_cfg=Loihi1SimCfg(select_tag="fixed_pt"))
                voltages_py[:, i] = dnf.v.get()
        finally:
            dnf.stop()

        ### LOIHI 2 ###

        gauss_pattern = GaussPattern(**gauss_pattern_params)
        spike_generator = RateCodeSpikeGen(**spike_gen_params)
        injector = PyToNxAdapter(shape=shape)
        dnf = LIF(**dnf_params)

        gauss_pattern.a_out.connect(spike_generator.a_in)
        spike_generator.s_out.connect(injector.inp)
        connect(injector.out, dnf.a_in, ops=[Weights(20)])
        connect(dnf.s_out, dnf.a_in, ops=[Convolution(kernel)])

        voltages_nc = np.zeros_like(voltages_py)
        try:
            for i in range(num_steps):
                dnf.run(condition=RunSteps(num_steps=1),
                        run_cfg=Loihi2HwCfg())
                voltages_nc[:, i] = dnf.v.get()
        finally:
            dnf.stop()

        voltages_nc = np.where(voltages_nc == 128, 0, voltages_nc)

        if verbose:
            print(f"{voltages_py=}")
            print(f"{voltages_nc=}")

        np.testing.assert_array_equal(voltages_py, voltages_nc)


if __name__ == '__main__':
    unittest.main()

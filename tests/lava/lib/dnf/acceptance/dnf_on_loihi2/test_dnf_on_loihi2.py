# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty
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
        self._compare_loihi_vs_python(shape=(7,))

    @unittest.skipUnless(run_loihi_tests, 'RUN_LOIHI_TESTS is not enabled.')
    def test_2d_dnf_on_loihi2(self) -> None:
        self._compare_loihi_vs_python(shape=(5, 5))

    @unittest.skipUnless(run_loihi_tests, 'RUN_LOIHI_TESTS is not enabled.')
    def test_3d_dnf_on_loihi2(self) -> None:
        self._compare_loihi_vs_python(shape=(3, 3, 3))

    def _compare_loihi_vs_python(self, shape: ty.Tuple[int, ...]) -> None:
        voltages_py = self._run_test(shape=shape, run_on_loihi=False)
        voltages_nc = self._run_test(shape=shape, run_on_loihi=True)

        if verbose:
            print(f"{voltages_py=}")
            print(f"{voltages_nc=}")

        np.testing.assert_array_equal(voltages_py, voltages_nc)

    @staticmethod
    def _run_test(shape: ty.Tuple[int, ...], run_on_loihi: bool) -> np.ndarray:
        num_steps = 10
        num_dimensions = len(shape)

        gauss_pattern = GaussPattern(shape=shape,
                                     amplitude=3000,
                                     mean=[1] * num_dimensions,
                                     stddev=[2] * num_dimensions)
        spike_generator = RateCodeSpikeGen(shape=shape, seed=1)
        dnf = LIF(shape=shape, du=409, dv=2047, vth=50)
        kernel = SelectiveKernel(amp_exc=18,
                                 width_exc=[3] * num_dimensions,
                                 global_inh=-15)

        gauss_pattern.a_out.connect(spike_generator.a_in)
        if run_on_loihi:
            injector = PyToNxAdapter(shape=shape)
            spike_generator.s_out.connect(injector.inp)
            connect(injector.out, dnf.a_in, ops=[Weights(20)])
        else:
            connect(spike_generator.s_out, dnf.a_in, ops=[Weights(20)])
        connect(dnf.s_out, dnf.a_in, ops=[Convolution(kernel)])

        run_cfg = Loihi2HwCfg() if run_on_loihi else Loihi1SimCfg(
            select_tag="fixed_pt")

        voltages = np.zeros(shape=shape + (num_steps,), dtype=int)
        try:
            for i in range(num_steps):
                dnf.run(condition=RunSteps(num_steps=1), run_cfg=run_cfg)
                voltages[..., i] = dnf.v.get()
        finally:
            dnf.stop()

        if run_on_loihi:
            voltages = np.where(voltages == 128, 0, voltages)

        return voltages


if __name__ == '__main__':
    unittest.main()

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen


class SinkProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, **kwargs: ty.Tuple[int, ...]) -> None:
        super().__init__(**kwargs)
        shape = kwargs.get("shape")

        self.data = Var(shape=shape, init=np.nan)

        self.s_in = InPort(shape=(shape[0],))


@implements(proc=SinkProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class SinkProcessModel(PyLoihiProcessModel):
    data: np.ndarray = LavaPyType(np.ndarray, float)

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)

    def run_spk(self) -> None:
        """Receive data and store in an internal variable"""
        s_in = self.s_in.recv()
        self.data[:, self.current_ts - 1] = s_in


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
        sink_process = SinkProcess(shape=(shape[0], num_steps_per_pattern * 2))

        gauss_pattern.out_ports.a_out.connect(spike_gen.in_ports.a_in)
        spike_gen.out_ports.s_out.connect(sink_process.in_ports.s_in)

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

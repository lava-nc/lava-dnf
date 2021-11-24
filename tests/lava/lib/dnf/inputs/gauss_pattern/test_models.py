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
from lava.lib.dnf.utils.math import gauss


class SinkProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, **kwargs: ty.Typle[int, ...]) -> None:
        super().__init__(**kwargs)
        shape = kwargs.get("shape")

        self.data = Var(shape=shape, init=0)

        self.a_in = InPort(shape=shape)


@implements(proc=SinkProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class SinkProcessModel(PyLoihiProcessModel):
    data: np.ndarray = LavaPyType(np.ndarray, float)

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        """Receive data and store in an internal variable"""
        # Receive data from PyInPort
        data = self.a_in.recv()

        # If the received pattern is not the null_pattern ...
        if not np.isnan(data).any():
            self.data = data


class TestGaussPatternProcessModel(unittest.TestCase):
    def test_gauss_pattern(self) -> None:
        """Tests whether GaussPatternProcessModel computes and sends a gauss
        pattern given its parameters."""
        gauss_pattern = GaussPattern(shape=(30, 30),
                                     amplitude=200.,
                                     mean=[15., 15.],
                                     stddev=[5., 5.])
        sink_process = SinkProcess(shape=(30, 30))
        gauss_pattern.out_ports.a_out.connect(sink_process.in_ports.a_in)

        gauss_generated_pattern = gauss(shape=(30, 30),
                                        domain=None,
                                        amplitude=200.,
                                        mean=np.array([15., 15.]),
                                        stddev=np.array([5., 5.]))

        try:
            gauss_pattern.run(condition=RunSteps(num_steps=3),
                              run_cfg=Loihi1SimCfg())

            np.testing.assert_array_equal(sink_process.data.get(),
                                          gauss_generated_pattern)
        finally:
            gauss_pattern.stop()

    def test_change_pattern_triggers_computation_and_send(self) -> None:
        """Tests whether GaussPatternProcessModel recomputes a new pattern and
        sends it when its parameters are changed. If that's the case, it will
        be received by the SinkProcess one timestep later"""
        gauss_pattern = GaussPattern(shape=(30, 30),
                                     amplitude=200.,
                                     mean=[15., 15.],
                                     stddev=[5., 5.])
        sink_process = SinkProcess(shape=(30, 30))
        gauss_pattern.out_ports.a_out.connect(sink_process.in_ports.a_in)

        gauss_generated_pattern = gauss(shape=(30, 30),
                                        domain=None,
                                        amplitude=100.,
                                        mean=np.array([10., 10.]),
                                        stddev=np.array([3., 3.]))

        try:
            gauss_pattern.run(condition=RunSteps(num_steps=3),
                              run_cfg=Loihi1SimCfg())

            gauss_pattern.amplitude = 100.
            gauss_pattern.mean = [10., 10.]
            gauss_pattern.stddev = [3., 3.]

            gauss_pattern.run(condition=RunSteps(num_steps=5),
                              run_cfg=Loihi1SimCfg())

            np.testing.assert_array_equal(sink_process.data.get(),
                                          gauss_generated_pattern)
        finally:
            gauss_pattern.stop()


if __name__ == '__main__':
    unittest.main()

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen


class SinkProcess(AbstractProcess):
    """
    Process that receives spike (bool) vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, **kwargs):
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

    def run_spk(self):
        """Receive data and store in an internal variable"""
        s_in = self.s_in.recv()
        self.data[:, self.current_ts - 1] = s_in


class SourceProcess(AbstractProcess):
    """
    Process that sends arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        data = kwargs.get("data")

        self.null_data = Var(shape=shape, init=np.full(shape, np.nan))

        self._data = Var(shape=shape, init=data)

        self.changed = Var(shape=(1,), init=True)

        self.a_out = OutPort(shape=shape)

    def _update(self):
        self.changed.set(np.array([True]))
        self.changed.get()

    @property
    def data(self):
        try:
            return self._data.get()
        except AttributeError:
            return None

    @data.setter
    def data(self, data):
        self._data.set(data)
        self._data.get()
        self._update()


@implements(proc=SourceProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class SourceProcessModel(PyLoihiProcessModel):
    null_data: np.ndarray = LavaPyType(np.ndarray, float)

    _data: np.ndarray = LavaPyType(np.ndarray, float)

    changed: np.ndarray = LavaPyType(np.ndarray, bool)

    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        """Send data when change is triggered, null_data otherwise"""
        if self.changed[0]:
            self.a_out.send(self._data)
            self.changed[0] = False
        else:
            self.a_out.send(self.null_data)


class TestRateCodeSpikeGenProcessModel(unittest.TestCase):
    def test_recv_null_pattern(self):
        """Tests that last_spiked, inter_spike_distances, first_spike_times
        Vars are not updated upon receipt of a null pattern."""
        pattern = np.zeros((30,))
        pattern[9:20] = 100.

        source = SourceProcess(shape=(30,), data=pattern)
        spike_generator = RateCodeSpikeGen(shape=(30,))

        source.out_ports.a_out.connect(spike_generator.in_ports.a_in)

        try:
            source.run(condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg())

            inter_spike_distances = spike_generator.inter_spike_distances.get()
            first_spike_times = spike_generator.first_spike_times.get()

            source.run(condition=RunSteps(num_steps=5), run_cfg=Loihi1SimCfg())

            np.testing.assert_array_equal(
                spike_generator.inter_spike_distances.get(),
                inter_spike_distances)

            np.testing.assert_array_equal(
                spike_generator.first_spike_times.get(),
                first_spike_times)
        finally:
            source.stop()

    def test_recv_non_null_pattern(self):
        """Tests whether last_spiked, inter_spike_distances,
        first_spike_times Vars are updated upon receipt of a new pattern."""
        pattern_1 = np.zeros((30,))
        pattern_1[9:20] = 100.

        pattern_2 = np.zeros((30,))
        pattern_2[15:25] = 150.

        source = SourceProcess(shape=(30,), data=pattern_1)
        spike_generator = RateCodeSpikeGen(shape=(30,), seed=42)

        source.out_ports.a_out.connect(spike_generator.in_ports.a_in)

        try:
            source.run(condition=RunSteps(num_steps=3), run_cfg=Loihi1SimCfg())

            old_inter_spike_distances = \
                spike_generator.inter_spike_distances.get()
            old_first_spike_times = \
                spike_generator.first_spike_times.get()

            source.data = pattern_2

            source.run(condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg())

            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(
                    spike_generator.inter_spike_distances.get(),
                    old_inter_spike_distances)

            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(
                    spike_generator.first_spike_times.get(),
                    old_first_spike_times)
        finally:
            source.stop()

    def test_compute_distances(self):
        """Tests whether inter spiked distances are computed correctly given
        a certain pattern."""
        pattern = np.zeros((30,))
        pattern[9:20] = 100.

        source = SourceProcess(shape=(30,), data=pattern)
        spike_generator = RateCodeSpikeGen(shape=(30,))

        source.out_ports.a_out.connect(spike_generator.in_ports.a_in)

        expected_inter_spike_distances = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            61.0, 61.0, 61.0, 61.0, 61.0, 61.0, 61.0, 61.0, 61.0, 61.0, 61.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

        try:
            source.run(condition=RunSteps(num_steps=2), run_cfg=Loihi1SimCfg())

            np.testing.assert_array_equal(
                spike_generator.inter_spike_distances.get(),
                np.array(expected_inter_spike_distances))
        finally:
            source.stop()

    def test_send(self):
        """Tests whether RateCodeSpikeGenProcessModel sends data through its
        OutPort every time step, regardless of whether its internal state
        (inter_spike_distances ...) changed or not."""
        num_steps = 10

        pattern = np.zeros((30,))
        pattern[9:20] = 100.

        source = SourceProcess(shape=(30,), data=pattern)
        spike_generator = RateCodeSpikeGen(shape=(30,))
        sink = SinkProcess(shape=(30, num_steps))

        source.out_ports.a_out.connect(spike_generator.in_ports.a_in)
        spike_generator.out_ports.s_out.connect(sink.in_ports.s_in)

        try:
            source.run(condition=RunSteps(num_steps=num_steps),
                       run_cfg=Loihi1SimCfg())

            self.assertFalse(np.isnan(sink.data.get()).any())
        finally:
            source.stop()

    def test_generate_spikes(self):
        """Tests whether the spike trains are computed correctly"""
        num_steps = 10

        pattern = np.zeros((20,))
        pattern[7:14] = 1500.

        expected_spike_trains = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        source = SourceProcess(shape=(20,), data=pattern)
        spike_generator = RateCodeSpikeGen(shape=(20,), seed=42)
        sink = SinkProcess(shape=(20, num_steps))

        source.out_ports.a_out.connect(spike_generator.in_ports.a_in)
        spike_generator.out_ports.s_out.connect(sink.in_ports.s_in)

        try:
            source.run(condition=RunSteps(num_steps=num_steps),
                       run_cfg=Loihi1SimCfg())

            np.testing.assert_array_equal(sink.data.get(),
                                          np.array(expected_spike_trains))
        finally:
            source.stop()


if __name__ == '__main__':
    unittest.main()

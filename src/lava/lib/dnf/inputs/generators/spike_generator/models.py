# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.lib.dnf.inputs.generators.spike_generator.process import SpikeGenerator

TIME_STEPS_PER_MINUTE = 6000.0
MIN_SPIKE_RATE = 0.5


@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class SpikeGeneratorProcessModel(PyLoihiProcessModel):
    inter_spike_distances: np.ndarray = LavaPyType(np.ndarray, int)
    first_spike_times: np.ndarray = LavaPyType(np.ndarray, int)
    last_spiked: np.ndarray = LavaPyType(np.ndarray, float)

    spikes: np.ndarray = LavaPyType(np.ndarray, bool)

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)

    ts_last_changed: int = 1

    def _compute_distances(self, pattern):
        distances = np.where(pattern > MIN_SPIKE_RATE, np.rint(TIME_STEPS_PER_MINUTE / pattern).astype(np.int), 0)

        return distances

    def _compute_first_spike_times(self, distances):
        rng = np.random.default_rng()

        first_spikes = np.zeros_like(distances)

        idx_zeros = distances == 0

        distances[~idx_zeros] = distances[~idx_zeros] + 1

        first_spikes[~idx_zeros] = rng.integers(low=np.ones_like(distances[~idx_zeros]), high=distances[~idx_zeros])

        return first_spikes

    def _generate_spikes(self,
                         time_step: int) -> np.ndarray:
        result = np.zeros(self.spikes.shape, np.bool)

        current_ts_transformed = time_step - self.ts_last_changed + 1

        idx_zeros = self.inter_spike_distances == np.zeros
        idx_never_spiked = self.last_spiked == - np.inf
        idx_will_spike_first_time = self.first_spike_times == current_ts_transformed

        distances_last_spiked = current_ts_transformed - self.last_spiked

        result[idx_will_spike_first_time] = True
        result[~idx_never_spiked] = \
            distances_last_spiked[~idx_never_spiked] == self.inter_spike_distances[~idx_never_spiked]
        result[idx_zeros] = False

        self.last_spiked[result] = current_ts_transformed

        return result

    def run_spk(self):
        if self.a_in.probe():
            self.ts_last_changed = self.current_ts
            self.last_spiked = np.full_like(self.last_spiked, -np.inf)

            pattern = self.a_in.recv()
            self.inter_spike_distances = self._compute_distances(pattern)
            self.first_spike_times = self._compute_first_spike_times(self.inter_spike_distances)

        self.spikes = self._generate_spikes(time_step=self.current_ts)
        self.s_out.send(self.spikes)

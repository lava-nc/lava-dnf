# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.lib.dnf.inputs.spike_generator.process import \
    SpikeGenerator

TIME_STEPS_PER_MINUTE = 6000.0
MIN_SPIKE_RATE = 0.5


# TODO: (GK) Change protocol to AsyncProtocol when supported (?)
# TODO: (GK) Change base class to (Sequential)PyProcessModel when supported (?)
@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class SpikeGeneratorProcessModel(PyLoihiProcessModel):
    """
    PyLoihiProcessModel for SpikeGeneratorProcess.

    Implements the behavior of a rate-coded spike input generator.
    """
    inter_spike_distances: np.ndarray = LavaPyType(np.ndarray, int)
    first_spike_times: np.ndarray = LavaPyType(np.ndarray, int)
    last_spiked: np.ndarray = LavaPyType(np.ndarray, float)

    spikes: np.ndarray = LavaPyType(np.ndarray, bool)

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)

    ts_last_changed: int = 1

    def _compute_distances(self, pattern: np.ndarray) -> np.ndarray:
        """Converts pattern representing spike rates in Hz to
        inter spike distances in timesteps.

        Assumes a minute contains TIME_STEPS_PER_MINUTE timesteps.
        Assumes all spike rate values less than MIN_SPIKE_RATE are negligible.

        Parameters
        ----------
        pattern : numpy.ndarray
            pattern representing spike rates in (Hz)

        Returns
        -------
        distances : numpy.ndarray
            inter spike distances in (timesteps)

        """
        # Represent infinite inter spike distances (i.e negligible spike rates)
        # as 0
        distances = np.where(pattern > MIN_SPIKE_RATE,
                             np.rint(TIME_STEPS_PER_MINUTE / pattern).astype(
                                 np.int), 0)

        return distances

    def _compute_first_spike_times(self, distances):
        """Randomly picks an array of first spike time given an array of
        inter spike distances.

        Every first spike time must be less than the associated
        inter spike distance.

        Parameters
        ----------
        distances : numpy.ndarray
            inter spike distances in (timesteps)

        Returns
        -------
        first_spike_times : numpy.ndarray
            first spike time for each "neuron"

        """
        rng = np.random.default_rng()

        first_spikes = np.zeros_like(distances)

        # Find indices where distance is 0 (where the should never be spikes)
        idx_zeros = distances == 0

        # Trick to yield a distances array in the right format for rng.integers
        distances[~idx_zeros] = distances[~idx_zeros] + 1

        # For indices where there should be a first spike ...
        # Pick a random number in [1, distance[
        first_spikes[~idx_zeros] = rng.integers(
            low=np.ones_like(distances[~idx_zeros]), high=distances[~idx_zeros])

        return first_spikes

    def _generate_spikes(self,
                         time_step: int) -> np.ndarray:
        """Generates an array of bool values where True represent a spike
        and False represents no-spike.

        Uses internal state such as inter spike distances, first spike times,
        last spike times and time step where last pattern change happened to
        derive whether each "neuron" should fire.

        Parameters
        ----------
        time_step : int
            current time step

        Returns
        -------
        spikes : numpy.ndarray
            spikes array

        """
        spikes = np.zeros(self.spikes.shape, np.bool)

        # Get time step index since pattern last changed
        current_ts_transformed = time_step - self.ts_last_changed + 1

        # Get indices where there should never be spikes
        idx_zeros = self.inter_spike_distances == np.zeros
        # Get indices where a spike never happened
        idx_never_spiked = self.last_spiked == - np.inf
        # Get indices where a first spike should be fired in this time step
        idx_will_spike_first_time = \
            self.first_spike_times == current_ts_transformed

        # Computes distances from current time step to last spike times
        distances_last_spiked = current_ts_transformed - self.last_spiked

        # Spike at indices where there should be a first spike
        spikes[idx_will_spike_first_time] = True
        # Spike at indices where we already spiked before, and where,
        # distance from current ts to last spike time is equal to
        # inter spike distance
        spikes[~idx_never_spiked] = \
            distances_last_spiked[~idx_never_spiked] == \
            self.inter_spike_distances[~idx_never_spiked]
        # Do not spike where there should never be spikes
        spikes[idx_zeros] = False

        # Update last spike times
        self.last_spiked[spikes] = current_ts_transformed

        return spikes

    def run_spk(self):
        # When a new pattern reached the PyInPort ...
        if self.a_in.probe():
            # Save the current time step
            self.ts_last_changed = self.current_ts
            # Reset last spike times
            self.last_spiked = np.full_like(self.last_spiked, -np.inf)

            # Receive the new pattern
            pattern = self.a_in.recv()
            # Update inter spike distances based on new pattern
            self.inter_spike_distances = self._compute_distances(pattern)
            # Compute first spike time for each "neuron"
            self.first_spike_times = self._compute_first_spike_times(
                self.inter_spike_distances)

        # Generate spike at every time step ...
        self.spikes = self._generate_spikes(time_step=self.current_ts)
        # ... and send them through the PyOutPort
        self.s_out.send(self.spikes)

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

from lava.lib.dnf.inputs.rate_code_spike_gen.process import \
    RateCodeSpikeGen

# TODO: (GK) This has to be changed to depend on time step duration in Loihi.
TIME_STEPS_PER_MINUTE = 6000.0


# TODO: (GK) Change protocol to AsyncProtocol when supported
# TODO: (GK) Change base class to (Sequential)PyProcessModel when supported
@implements(proc=RateCodeSpikeGen, protocol=LoihiProtocol)
@requires(CPU)
class RateCodeSpikeGenProcessModel(PyLoihiProcessModel):
    """
    PyLoihiProcessModel for SpikeGeneratorProcess.

    Implements the behavior of a rate-coded spike input generator.
    """
    min_spike_rate: np.ndarray = LavaPyType(np.ndarray, float)
    seed: np.ndarray = LavaPyType(np.ndarray, int)

    inter_spike_distances: np.ndarray = LavaPyType(np.ndarray, int)
    first_spike_times: np.ndarray = LavaPyType(np.ndarray, int)
    last_spiked: np.ndarray = LavaPyType(np.ndarray, float)

    spikes: np.ndarray = LavaPyType(np.ndarray, bool)

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.ts_last_changed: int = 1

    def run_spk(self) -> None:
        # Receive pattern from PyInPort
        pattern = self.a_in.recv()

        # If the received pattern is not the null_pattern ...
        if not np.isnan(pattern).any():
            # Reset time step counter.
            self.ts_last_changed = 1

            # Reset last spike times
            self.last_spiked[:] = -np.inf

            # Update inter spike distances based on new pattern
            self.inter_spike_distances = self._compute_spike_distances(pattern)
            # Compute first spike time for each "neuron"
            self.first_spike_times = self._compute_spike_onsets(
                self.inter_spike_distances)
        else:
            self.ts_last_changed += 1

        # Generate spike at every time step ...
        self.spikes = self._generate_spikes()
        # ... and send them through the PyOutPort
        self.s_out.send(self.spikes)

    def _compute_spike_distances(self, pattern: np.ndarray) -> np.ndarray:
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
        distances = np.zeros_like(pattern)

        idx_non_negligible = pattern > self.min_spike_rate[0]

        distances[idx_non_negligible] = \
            np.rint(TIME_STEPS_PER_MINUTE / pattern[idx_non_negligible])\
            .astype(int)

        idx_saturated = np.all([idx_non_negligible, distances == 0.], axis=0)

        distances[idx_saturated] = 1

        return distances

    def _compute_spike_onsets(self, distances: np.ndarray) -> np.ndarray:
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
        # Create a random number generator.
        seed = None if self.seed[0] == -1 else self.seed[0]
        rng = np.random.default_rng(seed=seed)

        first_spikes = np.zeros_like(distances)

        # Find indices where distance is 0 (where the should never be spikes)
        idx_zeros = distances == 0

        # For indices where there should be a first spike ...
        # Pick a random number in [1, distance[
        # (distance here is actually the true value of distance, +1)
        first_spikes[~idx_zeros] = rng.integers(
            low=np.ones_like(distances[~idx_zeros]),
            high=distances[~idx_zeros],
            endpoint=True)

        return first_spikes

    def _generate_spikes(self) -> np.ndarray:
        """Generates an array of bool values where True represent a spike
        and False represents no-spike.

        Uses internal state such as inter spike distances, first spike times,
        last spike times and time step where last pattern change happened to
        derive whether each "neuron" should fire.

        Returns
        -------
        spikes : numpy.ndarray
            spikes array

        """
        spikes = np.zeros(self.spikes.shape, dtype=bool)

        # Computes distances from current time step to last spike times
        distances_last_spiked = self.ts_last_changed - self.last_spiked

        # Get indices where a first spike should be fired in this time step.
        idx_will_spike_first_time = \
            self.first_spike_times == self.ts_last_changed
        # Spike at indices where there should be a first spike.
        spikes[idx_will_spike_first_time] = True

        # Get indices where a spike never happened.
        idx_never_spiked = self.last_spiked == - np.inf
        # Spike at indices where we already spiked before, and where,
        # distance from current ts to last spike time is equal to
        # inter spike distance.
        spikes[~idx_never_spiked] = \
            distances_last_spiked[~idx_never_spiked] == \
            self.inter_spike_distances[~idx_never_spiked]

        # Get indices where there should never be spikes
        idx_zeros = self.inter_spike_distances == np.zeros
        # Do not spike where there should never be spikes
        spikes[idx_zeros] = False

        # Update last spike times
        self.last_spiked[spikes] = self.ts_last_changed

        return spikes

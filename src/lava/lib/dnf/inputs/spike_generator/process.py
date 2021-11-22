# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.lib.dnf.utils.validation import validate_shape


# TODO: (GK) Should we name it RateSpikeGenerator ?
class SpikeGenerator(AbstractProcess):
    """
    Spike generator Process for rate-coded input.

    This process generates spike trains based on patterns it receives through
    its InPort a_in.
    It interprets these patterns as spiking rates (rate coding).

    Receives a new pattern through a_in only once and while and trigger state
    update upon receipt of new pattern.
    Sends spike values through its OutPort s_out every time step.

    Parameters:
    -----------
    shape: tuple(int)
        number of neurons per dimension, e.g. shape=(30, 40)
    min_spike_rate: float
        minimum spike rate
        (neurons with rates below this value will never spike)
    seed: int
        seed used for computing first spike times everytime pattern changes
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        shape = validate_shape(kwargs.pop("shape"))

        min_spike_rate = kwargs.pop("min_spike_rate", 0.5)
        if min_spike_rate < 0:
            raise ValueError("<min_spike_rate> cannot be negative.")

        # seed -1 means use random seed
        seed = kwargs.pop("seed", -1)
        if seed < -1:
            raise ValueError("<seed> cannot be negative.")

        self.min_spike_rate = Var(shape=(1,), init=np.array([min_spike_rate]))
        self.seed = Var(shape=(1,), init=np.array([seed]))

        self.inter_spike_distances = Var(shape=shape, init=np.zeros(shape))
        self.first_spike_times = Var(shape=shape, init=np.zeros(shape))
        self.last_spiked = Var(shape=shape, init=np.full(shape, -np.inf))

        self.spikes = Var(shape=shape, init=np.zeros(shape))

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

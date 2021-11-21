# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


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
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        shape = kwargs.pop("shape")

        self.inter_spike_distances = Var(shape=shape, init=0)
        self.first_spike_times = Var(shape=shape, init=0)
        self.last_spiked = Var(shape=shape, init=-np.inf)

        self.spikes = Var(shape=shape, init=0)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

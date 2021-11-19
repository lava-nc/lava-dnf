# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class SpikeGenerator(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        shape = kwargs.pop("shape")

        self.inter_spike_distances = Var(shape=shape, init=0)
        self.first_spike_times = Var(shape=shape, init=0)
        self.last_spiked = Var(shape=shape, init=-np.inf)

        self.spikes = Var(shape=shape, init=0)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

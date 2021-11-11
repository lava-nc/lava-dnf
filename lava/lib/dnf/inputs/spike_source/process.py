# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.lib.dnf.utils.validation import validate_shape

class SpikeSource(AbstractProcess):
    """
    Spike input generating process

    This process acts as a source for spike input.
    It has an OutPort through which it sends spike values (boolean) every timestep.

    Parameters:
    -----------
    shape: tuple(int) or int
        number of neurons per dimension, e.g. shape=(30, 40)
    spike_generator: SpikeInputGenerator
        spike generator that will generate booleans every timestep
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shape: tuple = validate_shape(kwargs.pop("shape", 1))

        self.s_out = OutPort(shape=self.shape)
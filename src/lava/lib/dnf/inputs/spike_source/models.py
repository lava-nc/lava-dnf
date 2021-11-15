# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.proc.lif.process import LIF
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires

from lava.lib.dnf.inputs.spike_source.process import SpikeSource
from lava.lib.dnf.inputs.inputs import SpikeInputGenerator

## Should implement AsyncProtocol instead of LoihiProtocol
## Should inherit from AbstractPyProcessModel (or other ?) instead of PyLoihiProcessModel
## Should overrride run method instead of run_spk
@implements(proc=SpikeSource, protocol=LoihiProtocol)
@requires(CPU)
class SpikeSourceProcessModel(PyLoihiProcessModel):
    """
    PyLoihiProcessModel for the SpikeSource Process.

    It holds a SpikeInputGenerator as instance variable and generates a vector of boolean (spike or no spike) at every timesteps.
    Spikes are sent through OutPorts to InPorts of other processes.

    Parameters
    ----------
    proc : SpikeSource
        An instance of the SpikeSource Process
    """
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.bool, precision=1)

    def __init__(self, proc: SpikeSource):
        self.spike_generator = proc.spike_generator

    def run_spk(self):
        generator_data = self.spike_generator.generate()
        self.s_out.send(generator_data)
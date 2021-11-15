# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.proc.lif.process import LIF
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires

from lava.lib.dnf.inputs.bias_source.process import BiasSource
from lava.lib.dnf.inputs.inputs import BiasInputGenerator

## Should implement AsyncProtocol instead of LoihiProtocol
## Should inherit from AbstractPyProcessModel (or other ?) instead of PyLoihiProcessModel
## Should overrride run method instead of run_spk
@implements(proc=BiasSource, protocol=LoihiProtocol)
@requires(CPU)
class BiasSourceProcessModel(PyLoihiProcessModel):
    """
    PyLoihiProcessModel for the BiasSrouce Process.

    It holds a BiasInputGenerator as instance variable and generates bias values at every timesteps.
    Bias values are sent through RefPorts to bias Variables of other processes.

    Parameters
    ----------
    proc : BiasSource
        An instance of the BiasSource Process
    """
    b_out: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.float)

    def __init__(self, proc: BiasSource):
        kwargs = proc.init_args

        generator: BiasInputGenerator = kwargs.pop("generator")

    def run_spk(self):
        pass # TODO

    
        
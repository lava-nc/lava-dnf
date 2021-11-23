# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.proc.lif.process import LIF
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements

from lava.lib.dnf.population.process import Population


@implements(proc=Population, protocol=LoihiProtocol)
class PopulationSubProcessModel(AbstractSubProcessModel):
    """
    SubProcessModel for the Population Process, which is only a wrapper
    around LIF neurons, providing a default set of parameters.

    Parameters
    ----------
    proc : Population
        An instance of the Population Process
    """
    def __init__(self, proc: Population):
        kwargs = proc.init_args
        dv: int = kwargs.pop("dv", 2047)
        du: int = kwargs.pop("du", 409)
        threshold: int = kwargs.pop("threshold", 200)
        bias: int = kwargs.pop("bias", 0)
        bias_exp: int = kwargs.pop("bias_exp", 1)

        # create LIF neurons
        self.neurons = LIF(name=proc.name + " LIF",
                           shape=proc.shape,
                           vth=threshold,
                           du=du,
                           dv=dv,
                           bias=bias,
                           bias_exp=bias_exp)

        # expose the input and output port of the LIF process
        proc.in_ports.a_in.connect(self.neurons.in_ports.a_in)
        self.neurons.out_ports.s_out.connect(proc.out_ports.s_out)

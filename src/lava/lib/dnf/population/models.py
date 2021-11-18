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
        tau_voltage: int = kwargs.pop("tau_voltage", 2)
        tau_current: int = kwargs.pop("tau_current", 10)
        threshold: int = kwargs.pop("threshold", 200)
        bias: int = kwargs.pop("bias", 0)
        bias_exp: int = kwargs.pop("bias_exp", 1)

        # create LIF neurons
        self.neurons = LIF(name=proc.name + " LIF",
                           shape=proc.shape,
                           vth=threshold,
                           du=self._tau_to_decay(tau_current),
                           dv=self._tau_to_decay(tau_voltage),
                           bias_mant=bias,
                           bias_exp=bias_exp,
                           delay_bits=1)

        # expose the input and output port of the LIF process
        proc.in_ports.a_in.connect(self.neurons.in_ports.a_in)
        self.neurons.out_ports.s_out.connect(proc.out_ports.s_out)

    @staticmethod
    def _tau_to_decay(tau: ty.Union[int, float]) -> int:
        """
        Converts a time scale parameter (tau) to a decay value.

        Parameters:
        -----------
        tau : int or float
            time scale of a neural dynamics

        Returns:
        --------
        decay : int
            decay that corresponds to the given time scale parameter
        """
        if not tau >= 1:
            raise ValueError("tau must be greater-equal 1")

        return int(4095 / tau)

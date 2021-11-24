# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.lib.dnf.utils.validation import validate_shape


class Population(AbstractProcess):
    """
    Population of leaky integrate-and-fire (LIF) neurons

    This process is a wrapper around the LIF process.
    It only exposes a small subset of the LIF parameters and otherwise sets
    default parameters.

    Parameters:
    -----------
    shape: tuple(int) or int
        number of neurons per dimension, e.g. shape=(30, 40)
    tau_voltage : int
        time scale of the LIF voltage (internally converted to voltage decay)
    tau_current: int
        time scale of the LIF current (internally converted to current decay)
    threshold: int
        threshold of the LIF neurons
    bias_mant: int
        mantissa of the LIF bias
    bias_exp: int
        exponent of the LIF bias
    """
    def __init__(self,
                 **kwargs: ty.Union[str, int, ty.Tuple[int, ...]]) -> None:
        super().__init__(**kwargs)

        self.shape: tuple = validate_shape(kwargs.pop("shape", 1))

        self.a_in = InPort(shape=self.shape)
        self.s_out = OutPort(shape=self.shape)

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import RefPort

from lava.lib.dnf.utils.validation import validate_shape

class BiasSource(AbstractProcess):
    """
    Bias input generating process

    This process acts as a source for bias input.
    It has an RefPort through which it sends bias values (floating point numbers) every timestep.

    Parameters:
    -----------
    shape: tuple(int) or int
        number of neurons per dimension, e.g. shape=(30, 40)
    bias_generator: BiasInputGenerator
        bias generator that will generate floating point numbers every timestep
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shape: tuple = validate_shape(kwargs.pop("shape", 1))

        self.b_out = RefPort(shape=self.shape)
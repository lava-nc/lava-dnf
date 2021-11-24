# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var

from lava.lib.dnf.utils.validation import validate_shape


class ReshapeBool(AbstractProcess):
    """
    Reshapes the input to the output shape, keeping the number of elements
    constant.

    TODO (MR): Workaround in the absence of Reshape ports.

    Parameters:
    -----------
    shape_in: tuple(int) or int
        input shape
    shape_out: tuple(int) or int
        output shape
    """
    def __init__(self, **kwargs: ty.Tuple[int, ...]) -> None:
        super().__init__(**kwargs)

        shape_in = validate_shape(kwargs.pop("shape_in", (1,)))
        shape_out = validate_shape(kwargs.pop("shape_out", (1,)))
        shape_out_array = np.array(shape_out)
        self.shape_out = Var(shape=shape_out_array.shape, init=shape_out_array)

        self.s_in = InPort(shape=shape_in)
        self.s_out = OutPort(shape=shape_out)

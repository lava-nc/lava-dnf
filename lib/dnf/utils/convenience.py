# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


def num_neurons(shape: ty.Tuple[int, ...]) -> int:
    """
    Computes the number of neurons from a shape.

    Parameters:
    -----------
    shape : tuple(int)
        shape of a neural population (or input)

    Returns:
    --------
    num_neurons : int
        number of neurons
    """
    return int(np.prod(shape))

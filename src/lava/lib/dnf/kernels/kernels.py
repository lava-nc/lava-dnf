# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


class Kernel:
    """
    Represents a kernel that can be used in the Convolution operation.

    Parameters
    ----------
    weights : numpy.ndarray
        weight matrix of the kernel
    padding_value : float, optional
        value that is used to pad the kernel when the Convolution operation
        uses BorderType.PADDED
    """
    def __init__(self,
                 weights: np.ndarray,
                 padding_value: ty.Optional[float] = 0):
        self._weights = weights
        self._padding_value = padding_value

    @property
    def weights(self) -> np.ndarray:
        """Returns the weights"""
        return self._weights

    @property
    def padding_value(self) -> float:
        """Returns the padding value"""
        return self._padding_value

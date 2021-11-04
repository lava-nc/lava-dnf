# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod
import typing as ty
import numpy as np


class AbstractOperation(ABC):
    def __init__(self):
        self.changes_dim = False
        self.changes_size = False
        self.reorders_shape = False

        self.input_shape = None
        self.output_shape = None

        self._is_configured = False

    def configure(self,
                  input_shape: ty.Tuple[int, ...],
                  output_shape: ty.Tuple[int, ...]):
        self._configure(input_shape, output_shape)
        valid_configuration = self._validate_configuration()

        if valid_configuration:
            self._is_configured = True
        else:
            raise ValueError("configuration of operation is invalid; check "
                             "input_shape and output_shape")

    def compute_weights(self) -> np.ndarray:
        if not self._is_configured:
            raise AssertionError("operation must be configured before "
                                 "computing_weights() is called")
        else:
            return self._compute_weights()

    def _configure(self,
                   input_shape: ty.Tuple[int, ...],
                   output_shape: ty.Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def _validate_configuration(self) -> bool:
        pass

    @abstractmethod
    def _compute_weights(self) -> np.ndarray:
        pass

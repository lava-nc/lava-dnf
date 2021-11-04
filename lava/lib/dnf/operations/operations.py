# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod
import typing as ty
import numpy as np

from lava.lib.dnf.utils.convenience import num_neurons
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError


class AbstractOperation(ABC):
    """
    Abstract class for operations that can be used to parameterize the
    connect() function. An operation has an input shape and an output shape
    and, depending on the type of operation, will make changes to the
    dimensionality of the shape (e.g., from input shape (40, 20) to output
    shape (40,)), to the order of elements in the shape (e.g., from input
    shape (40, 20) to output shape (20, 40)), to the size of the shape (e.g.,
    from input shape (40, 20) to output shape (35, 15), or a combination of
    these changes. Every operation must mark the type of changes it makes by
    setting the attributes 'changes_dim', 'reorders_shape',
    and 'changes_size', respectively.

    """
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
        """
        Configures an operation by setting its input and output shape and
        validating that configuration.

        Parameters
        ----------
        input_shape : tuple(int)
            input shape of the operation
        output_shape : tuple(int)
            output shape of the operation

        """
        self._configure(input_shape, output_shape)

        # check that a basic configuration has been set
        if self.input_shape is None or self.output_shape is None:
            raise AssertionError("<input_shape> and <output_shape> "
                                 "may not be None")

        self._validate_configuration()
        self._is_configured = True

    def compute_weights(self) -> np.ndarray:
        """
        Computes the connectivity weight matrix of the operation.

        Returns
        -------
        connectivity weight matrix : numpy.ndarray

        """
        if not self._is_configured:
            raise AssertionError("operation must be configured before "
                                 "computing_weights() is called")
        else:
            return self._compute_weights()

    def _configure(self,
                   input_shape: ty.Tuple[int, ...],
                   output_shape: ty.Tuple[int, ...]):
        """
        Does the actual work of configuration; this method should be overwritten
        in subclasses if so desired.

        Parameters
        ----------
        input_shape : tuple(int)
            input shape of the operation
        output_shape : tuple(int)
            output shape of the operation

        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def _validate_configuration(self):
        """
        Validates the configuration (input_shape, output_shape) of the
        operation. Should raise a MisconfiguredOpError if the configuration
        is invalid.

        """
        pass

    @abstractmethod
    def _compute_weights(self) -> np.ndarray:
        """
        Does the actual work of computing the weights and returns it as a
        numpy array.

        Returns
        -------
        weights : numpy.ndarray

        """
        pass


class Weights(AbstractOperation):
    """
    Operation that generates one-to-one connectivity with given weights for
    every synapse.

    Parameters
    ----------
    weight : float
        weight used for every connection

    """
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def _compute_weights(self) -> np.ndarray:
        return np.eye(num_neurons(self.output_shape),
                      num_neurons(self.input_shape),
                      dtype=np.int32) * self.weight

    def _validate_configuration(self):
        # check that input and output shape matches
        if self.input_shape != self.output_shape:
            raise MisconfiguredOpError("Input and output shape must match")

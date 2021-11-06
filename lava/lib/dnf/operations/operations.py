# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, ABCMeta, abstractmethod
import typing as ty
import numpy as np

from lava.lib.dnf.utils.convenience import num_neurons
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError


class AbstractOperation(ABC):
    """
    Abstract Operation, subclasses of which can be used to parameterize the
    connect() function.

    """
    def __init__(self):
        self._input_shape = None
        self._output_shape = None

    @property
    def output_shape(self) -> ty.Tuple[int, ...]:
        """Return the output shape of the operation"""
        return self._output_shape

    def compute_weights(self) -> np.ndarray:
        """
        Computes the connectivity weight matrix of the operation.
        This public method only validates the configuration of the
        operation. The actual weights are computed in _compute_weights(),
        which must be implemented by any concrete subclass.

        Returns
        -------
        connectivity weight matrix : numpy.ndarray

        """
        # check that a basic configuration has been set
        if self._input_shape is None or self._output_shape is None:
            raise AssertionError("_input_shape and _output_shape "
                                 "should not be None; make sure to set "
                                 "_output_shape in _configure()")

        # compute and return connectivity weight matrix
        return self._compute_weights()

    @abstractmethod
    def _compute_weights(self) -> np.ndarray:
        """
        Does the actual work of computing the weights and returns them as a
        numpy array.

        Returns
        -------
        weights : numpy.ndarray

        """
        pass

    @abstractmethod
    def configure(self,
                  input_shape: ty.Tuple[int, ...]):
        """
        Configures an operation by setting its input and output shape.

        Parameters
        ----------
        input_shape : tuple(int)
            input shape of the operation

        """
        pass


class AbstractComputedShapeOperation(AbstractOperation):
    """
    Abstract Operation class for operations whose output shape can be derived
    from its input shape (potentially with the help of additional parameters).

    """
    def configure(self,
                  input_shape: ty.Tuple[int, ...]):
        self._input_shape = input_shape
        self._compute_output_shape(input_shape)

    @abstractmethod
    def _compute_output_shape(self, input_shape: ty.Tuple[int, ...]):
        """
        Derives the output shape of the operation from the input shape.

        Parameters
        ----------
        input_shape : tuple(int)
            input shape of the operation

        """
        pass


class AbstractSpecifiedShapeOperation(AbstractOperation):
    """
    Abstract Operation class for operations whose output shape must be
    specified by the user.

    Parameters
    ----------
    output_shape : tuple(int)
        output shape of the operation

    """
    def __init__(self, output_shape: ty.Tuple[int, ...]):
        super().__init__()
        self._output_shape = output_shape

    def configure(self,
                  input_shape: ty.Tuple[int, ...]):
        self._input_shape = input_shape
        self._validate_output_shape()

    @abstractmethod
    def _validate_output_shape(self):
        """
        Validates the output shape of the operation given the input shape, the
        purpose of the operation, as well as any other user inputs and
        constraints.

        Should raise an exception if the output shape is not valid.
        """
        pass


class AbstractKeepShapeOperation(AbstractComputedShapeOperation,
                                 metaclass=ABCMeta):
    """Abstract Operation that does not change the shape of the input."""
    def _compute_output_shape(self, input_shape: ty.Tuple[int, ...]):
        self._output_shape = input_shape


class AbstractReduceDimsOperation(AbstractComputedShapeOperation,
                                  metaclass=ABCMeta):
    """
    Abstract Operation that (only) reduces the dimensionality of the
    input.

    Parameters
    ----------
    reduce_dims : int or tuple(int)
        indices of the dimensions to remove
    """
    def __init__(self,
                 reduce_dims: ty.Union[int, ty.Tuple[int, ...]]):
        super().__init__()
        if isinstance(reduce_dims, int):
            reduce_dims = (reduce_dims,)
        self.reduce_dims = reduce_dims

    def _compute_output_shape(self, input_shape: ty.Tuple[int, ...]):
        self._validate_reduce_dims()
        self._output_shape = tuple(np.delete(np.asarray(input_shape),
                                             self.reduce_dims))

    def _validate_reduce_dims(self):
        if len(self.reduce_dims) == 0:
            raise ValueError("<reduce_dims> may not be empty")

        for idx in self.reduce_dims:
            # compute the positive index irrespective of the sign of 'idx'
            idx_pos = len(self._input_shape) + idx if idx < 0 else idx
            # make sure the positive index is not out of bounds
            if 0 < idx_pos >= len(self._input_shape):
                raise IndexError(f"<reduce_dims> value {idx} is out of bounds "
                                 f"for array of size {len(self._input_shape)}")


class AbstractReshapeOperation(AbstractSpecifiedShapeOperation,
                               metaclass=ABCMeta):
    """Abstract Operation that reshapes the input, changing the shape but
    keeping the number of elements constant."""
    def __init__(self, output_shape):
        super().__init__(output_shape=output_shape)

    def _validate_output_shape(self):
        if num_neurons(self._input_shape) != num_neurons(self._output_shape):
            raise MisconfiguredOpError("input and output shape must have the "
                                       "same number of elements")


class Weights(AbstractKeepShapeOperation):
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
        return np.eye(num_neurons(self._output_shape),
                      num_neurons(self._input_shape),
                      dtype=np.int32) * self.weight

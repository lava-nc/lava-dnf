# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod
import typing as ty
import numpy as np

from lava.lib.dnf.utils.convenience import num_neurons, num_dims
from lava.lib.dnf.utils.math import is_odd
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError


class AbstractShapeHandler(ABC):
    """
    Abstract class for handling input and output shape of the
    AbstractOperation class.

    """
    def __init__(self) -> None:
        self._input_shape = None
        self._output_shape = None

    def configure(self,
                  input_shape: ty.Tuple[int, ...]) -> None:
        """
        Configures the input and output shape of an operation given an input
        shape.

        Parameters
        ----------
        input_shape : tuple(int)
            input shape of an operation

        """
        self._validate_input_shape(input_shape)
        self._input_shape = input_shape
        # Validate any arguments that subclass shape handlers may receive
        self._validate_args()
        self._compute_output_shape()

    def assert_configured(self) -> None:
        """Assert that input and output shape is configured."""
        if self._input_shape is None or self._output_shape is None:
            raise AssertionError("_input_shape and _output_shape "
                                 "should not be None")

    @property
    def output_shape(self) -> ty.Tuple[int, ...]:
        """Return the output shape of the handler"""
        return self._output_shape

    @property
    def input_shape(self) -> ty.Tuple[int, ...]:
        """Return the input shape of the handler"""
        return self._input_shape

    @abstractmethod
    def _compute_output_shape(self) -> None:
        pass

    @abstractmethod
    def _validate_args(self) -> None:
        pass

    @abstractmethod
    def _validate_input_shape(self,
                              input_shape: ty.Tuple[int, ...]) -> None:
        pass


class KeepShapeShapeHandler(AbstractShapeHandler):
    """Shape handler for operations that do not change the shape of the
     input."""
    def _compute_output_shape(self) -> None:
        self._output_shape = self._input_shape

    def _validate_args(self) -> None:
        pass

    def _validate_input_shape(self,
                              input_shape: ty.Tuple[int, ...]) -> None:
        pass


class ReduceDimsShapeHandler(AbstractShapeHandler):
    """
    Shape handler for operations that reduce the dimensionality of the
    input.

    Parameters
    ----------
    reduce_dims : int or tuple(int)
        indices of the dimensions to remove
    """
    def __init__(self,
                 reduce_dims: ty.Union[int, ty.Tuple[int, ...]]) -> None:
        super().__init__()
        if isinstance(reduce_dims, int):
            reduce_dims = (reduce_dims,)
        self._reduce_dims = reduce_dims

    @property
    def reduce_dims(self) -> ty.Tuple[int, ...]:
        """Return the output shape of the handler"""
        return self._reduce_dims

    def _compute_output_shape(self) -> None:
        self._output_shape = tuple(np.delete(np.asarray(self._input_shape),
                                             self.reduce_dims))
        if self._output_shape == ():
            self._output_shape = (1,)

    def _validate_input_shape(self, input_shape: ty.Tuple[int, ...]) -> None:
        if num_dims(input_shape) == 0:
            raise MisconfiguredOpError("ReduceDims shape handler is "
                                       "configured with an input shape that "
                                       "is already zero-dimensional")

    def _validate_args(self) -> None:
        """Validate the <reduce_dims> argument"""
        if len(self.reduce_dims) == 0:
            raise ValueError("<reduce_dims> may not be empty")

        if len(self.reduce_dims) > len(self._input_shape):
            raise ValueError(f"given <reduce_dims> {self.reduce_dims} has "
                             f"more entries than the shape of the input "
                             f"{self._input_shape}")

        _check_index_bounds(self.reduce_dims, self._input_shape)


class ExpandDimsShapeHandler(AbstractShapeHandler):
    """Shape handler for operations that expand the dimensionality. New
    dimensions (axes) will be appended to the already existing ones of the
    input. Their sizes must be specified using the <new_dims_shape> argument.

    Parameters
    ----------
    new_dims_shape : int or tuple(int)
        shape of the added dimensions; they will be appended to the shape
        of the input, for instance an input shape (2,) and
        new_dims_shape=(6, 8) will produce an output shape (2, 6, 8)
    """
    def __init__(self,
                 new_dims_shape: ty.Union[int, ty.Tuple[int, ...]]) -> None:
        super().__init__()
        if isinstance(new_dims_shape, int):
            new_dims_shape = (new_dims_shape,)
        self._new_dims_shape = new_dims_shape

    @property
    def new_dims_shape(self) -> ty.Tuple[int, ...]:
        """Return the <new_dims_shape> attribute"""
        return self._new_dims_shape

    def _compute_output_shape(self) -> None:
        if num_dims(self.input_shape) == 0:
            self._output_shape = self.new_dims_shape
        else:
            self._output_shape = self.input_shape + self.new_dims_shape

        if len(self._output_shape) > 3:
            raise NotImplementedError("ExpandDims operation is configured to "
                                      "produce an output shape with "
                                      "dimensionality larger than 3; higher "
                                      "dimensionality is currently not "
                                      "supported")

    def _validate_args(self) -> None:
        """Validate the <new_dims_shape> argument"""
        if len(self.new_dims_shape) == 0:
            raise ValueError("<new_dims_shape> may not be empty")

        if any(s < 1 for s in self.new_dims_shape):
            raise ValueError("values in <new_dims_shape> may not be smaller "
                             "than 1")

    def _validate_input_shape(self, input_shape: ty.Tuple[int, ...]) -> None:
        pass


class ReshapeShapeHandler(AbstractShapeHandler):
    """Shape handler for operations that reshape the input, changing
    the shape but keeping the number of elements constant.

    Parameters
    ----------
    output_shape : tuple(int)
        output shape of an operation

    """
    def __init__(self, output_shape: ty.Tuple[int, ...]) -> None:
        super().__init__()
        self._output_shape = output_shape

    def _validate_args(self) -> None:
        if num_neurons(self._input_shape) != num_neurons(self._output_shape):
            raise MisconfiguredOpError("input and output shape must have the "
                                       "same number of elements")

    def _compute_output_shape(self) -> None:
        pass

    def _validate_input_shape(self, input_shape: ty.Tuple[int, ...]) -> None:
        pass


class ReorderShapeHandler(AbstractShapeHandler):
    """Shape handler for operations that reorder the input shape.

    Parameters
    ----------
    order : tuple(int)
        order of the dimensions of the output; for instance if the input shape
        is (1, 2, 3) and order=(0, 2, 1), the output shape will be (1, 3, 2);
        must have the same number of elements as the input and output shape

    """
    def __init__(self, order: ty.Tuple[int, ...]) -> None:
        super().__init__()
        self._order = order

    @property
    def order(self) -> ty.Tuple[int, ...]:
        """Return the order of the handler"""
        return self._order

    def _compute_output_shape(self) -> None:
        input_shape = np.array(self._input_shape)
        self._output_shape = tuple(input_shape[list(self._order)])

    def _validate_args(self) -> None:
        """Validate the <order> argument"""
        num_dims_in = num_dims(self._input_shape)

        if len(self._order) != num_dims_in:
            raise MisconfiguredOpError("<order> must have the same number of "
                                       "entries as the input shape: "
                                       f"len({self._order}) != len("
                                       f"{self._input_shape})")

        _check_index_bounds(self._order, self._input_shape)

    def _validate_input_shape(self, input_shape: ty.Tuple[int, ...]) -> None:
        num_dims_in = num_dims(input_shape)

        if num_dims_in < 2:
            raise MisconfiguredOpError("the input dimensionality "
                                       "is smaller than 2; there are no "
                                       "dimensions to reorder")


class ReduceDiagonalShapeHandler(AbstractShapeHandler):
    """Shape handler for the ReduceDiagonal operation that projects
    diagonally from a multi-dimensional population.
    """
    def __init__(self) -> None:
        super().__init__()

    def _compute_output_shape(self) -> None:
        num_dims_in = num_dims(self._input_shape)
        shape = np.array(self._input_shape[0:int(num_dims_in/2)])
        self._output_shape = tuple(shape * 2 - 1)

    def _validate_args(self) -> None:
        pass

    def _validate_input_shape(self, input_shape: ty.Tuple[int, ...]) -> None:
        num_dims_in = num_dims(input_shape)

        if is_odd(num_dims_in):
            raise MisconfiguredOpError("the input dimensionality must be even")

        first_half = input_shape[0:int(num_dims_in/2)]
        second_half = input_shape[int(num_dims_in/2):]

        if first_half != second_half:
            raise MisconfiguredOpError(
                f"the first half of the input shape {first_half} must be "
                f"identical to the second half {second_half}")


class FlipShapeHandler(KeepShapeShapeHandler):
    """Shape handler for the Flip operation that flips specified dimensions.

    Parameters
    ----------
    dims : tuple(int) or int
        indices of the dimensions that are to be flipped
    """
    def __init__(
        self,
        dims: ty.Optional[ty.Union[int, ty.Tuple[int, ...]]] = None
    ) -> None:
        super().__init__()
        self._dims = dims

    @property
    def dims(self) -> ty.Tuple[int, ...]:
        """Return the <dims> that are to be flipped"""
        return self._dims

    def _validate_args(self) -> None:
        """Validate the <dims> argument"""
        num_dims_in = num_dims(self._input_shape)

        if self._dims is None:
            self._dims = tuple(range(num_dims_in))

        if len(self._dims) > num_dims_in:
            raise MisconfiguredOpError("<dims> may only have as many entries "
                                       "as the input shape: "
                                       f"len({self._dims}) > len("
                                       f"{self._input_shape})")

        _check_index_bounds(self._dims, self._input_shape)


def _check_index_bounds(indices: ty.Tuple[int, ...],
                        shape: ty.Tuple[int, ...]) -> None:
    """Checks whether all of the given indices are not out of bounds for the
    given shape. Throws an IndexError if violated.

    Parameters
    ----------
    indices : tuple(int)
        indices of the dimensions that are to be checked
    shape : tuple(int)
        shape of the array that the indices will address and that will be
        checked against

    """
    for idx in indices:
        # Compute the positive index irrespective of the sign of 'idx'
        idx_positive = len(shape) + idx if idx < 0 else idx
        # Make sure the positive index is not out of bounds
        if idx_positive < 0 or idx_positive >= len(shape):
            raise IndexError(f"index value {idx} is out of bounds "
                             f"for array of size {len(shape)}")

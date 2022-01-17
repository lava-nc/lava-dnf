# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod
import typing as ty
import numpy as np

from lava.lib.dnf.utils.convenience import num_neurons
from lava.lib.dnf.operations.shape_handlers import (
    AbstractShapeHandler,
    KeepShapeHandler,
    ReduceDimsHandler,
    ExpandDimsHandler,
    ReorderHandler,
    ReduceDiagonalHandler)
from lava.lib.dnf.operations.enums import ReduceMethod, BorderType
from lava.lib.dnf.kernels.kernels import Kernel
from lava.lib.dnf.utils.convenience import num_dims
from lava.lib.dnf.utils.math import is_odd


class AbstractOperation(ABC):
    """
    Abstract Operation, subclasses of which can be used to parameterize the
    connect() function.

    Parameters
    ----------
    shape_handler : AbstractShapeHandler
        handles, configures, and validates the input and output shape of the
        operation

    """
    def __init__(self, shape_handler: AbstractShapeHandler) -> None:
        self._shape_handler = shape_handler

    @property
    def output_shape(self) -> ty.Tuple[int, ...]:
        """Return the output shape of the operation"""
        return self._shape_handler.output_shape

    @property
    def input_shape(self) -> ty.Tuple[int, ...]:
        """Return the output shape of the operation"""
        return self._shape_handler.input_shape

    def compute_weights(self) -> np.ndarray:
        """
        Computes the connectivity weight matrix of the operation.
        This public method only validates the configuration of the
        operation. The actual weights are computed in the
        abstract method _compute_weights().

        Returns
        -------
        connectivity weight matrix : numpy.ndarray

        """
        # Assert that the input and output shape is configured
        self._shape_handler.assert_configured()

        return self._compute_weights()

    def configure(self,
                  input_shape: ty.Tuple[int, ...]) -> None:
        """
        Configures an operation by setting its input and output shape.

        Parameters
        ----------
        input_shape : tuple(int)
            input shape of the operation

        """
        self._validate_args_with_input_shape(input_shape)
        self._shape_handler.configure(input_shape)

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

    def _validate_args_with_input_shape(
        self,
        input_shape: ty.Tuple[int, ...]
    ) -> None:
        """Validates any input arguments that the operation may receive, and
        that do not get passed on to the ShapeHandler, against the input
        shape."""
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
    def __init__(self, weight: float) -> None:
        super().__init__(KeepShapeHandler())
        self.weight = weight

    def _compute_weights(self) -> np.ndarray:
        return np.eye(num_neurons(self.output_shape),
                      num_neurons(self.input_shape),
                      dtype=np.int32) * self.weight


class ReduceDims(AbstractOperation):
    """
    Operation that reduces the dimensionality of the input by projecting
    a specified subset of dimensions onto the remaining dimensions.

    Parameters
    ----------
    reduce_dims : int or tuple(int)
        indices of dimension that will be reduced/removed
    reduce_method : ReduceMethod
        method by which the dimensions will be reduced (SUM or MEAN)

    """
    def __init__(self,
                 reduce_dims: ty.Union[int, ty.Tuple[int, ...]],
                 reduce_method: ty.Optional[ReduceMethod] = ReduceMethod.SUM
                 ) -> None:
        super().__init__(ReduceDimsHandler(reduce_dims))
        ReduceMethod.validate(reduce_method)
        self.reduce_method = reduce_method

    def _compute_weights(self) -> np.ndarray:
        # Indices of the input dimensions in the weight matrix
        # that will not be removed
        in_axes_all = np.arange(num_dims(self.input_shape))

        sh = ty.cast(ReduceDimsHandler, self._shape_handler)
        in_axes_kept = tuple(np.delete(in_axes_all, sh.reduce_dims))

        # Generate the weight matrix
        weights = _project_dims(self.input_shape,
                                self.output_shape,
                                in_axes_kept=in_axes_kept)

        if self.reduce_method == ReduceMethod.MEAN:
            # Set the weights such that they compute the mean
            weights = weights / num_neurons(self.input_shape)

        return weights


class ExpandDims(AbstractOperation):
    """
    Operation that expands the dimensionality of the input by projecting
    the dimensions of the input to the newly added dimensions.

    """
    def __init__(self,
                 new_dims_shape: ty.Union[int, ty.Tuple[int, ...]]) -> None:
        super().__init__(ExpandDimsHandler(new_dims_shape))

    def _compute_weights(self) -> np.ndarray:
        # Indices of the output dimensions in the weight matrix that will
        # be kept from the input
        out_axes_kept = tuple(np.arange(num_dims(self.input_shape)))

        # Generate the weight matrix
        weights = _project_dims(self.input_shape,
                                self.output_shape,
                                out_axes_kept=out_axes_kept)

        return weights


class Reorder(AbstractOperation):
    """
    Operation that reorders the dimensions in the input to a specified new
    order.

    Parameters
    ----------
    order : tuple(int)
        new order of the dimensions (see ReorderHandler)

    """
    def __init__(self, order: ty.Tuple[int, ...]) -> None:
        super().__init__(ReorderHandler(order))

    def _compute_weights(self) -> np.ndarray:
        sh = ty.cast(ReorderHandler,
                     self._shape_handler)
        weights = _project_dims(self.input_shape,
                                self.output_shape,
                                out_axes_kept=sh.order)

        return weights


def _project_dims(
    input_shape: ty.Tuple[int, ...],
    output_shape: ty.Tuple[int, ...],
    out_axes_kept: ty.Optional[ty.Tuple[int, ...]] = None,
    in_axes_kept: ty.Optional[ty.Tuple[int, ...]] = None
) -> np.ndarray:
    """Projection function that is used both by the ReduceDims and ExpandDims
    Operation

    Parameters
    ----------
    input_shape : tuple(int)
        input shape of the operation
    output_shape : tuple(int)
        output shape of the operation
    out_axes_kept : tuple(int)
        indices of the output dimensions in the weight matrix that will
        be kept from the input
    in_axes_kept : tuple(int)
        indices of the input dimensions in the weight matrix that will
        be kept for the output

    Returns
    -------
    connectivity weight matrix : numpy.ndarray

    """
    num_neurons_in = num_neurons(input_shape)
    num_neurons_out = num_neurons(output_shape)
    num_dims_in = num_dims(input_shape)
    num_dims_out = num_dims(output_shape)
    smaller_num_dims = min(num_dims_in, num_dims_out)

    if smaller_num_dims == 0:
        # If the target is a 0D population, the connectivity is from
        # all neurons in the source population to that one neuron
        weights = np.ones((num_neurons_out, num_neurons_in))
    else:
        # Create a dense connectivity matrix, where dimensions of the
        # source and target are not yet flattened
        shape = output_shape + input_shape
        weights = np.zeros(shape)

        ###
        # The following lines create a view on the connectivity matrix,
        # in which the axes are moved such that the first dimensions are all
        # output dimensions that will be kept, followed by all input
        # dimensions that will be kept, followed by all remaining dimensions.
        if in_axes_kept is None:
            in_axes_kept = np.arange(num_dims_in)
        in_axes_kept = tuple(np.asarray(in_axes_kept) + num_dims_out)

        if out_axes_kept is None:
            out_axes_kept = np.arange(num_dims_out)
        out_axes_kept = tuple(out_axes_kept)

        # New indices of the kept output dimensions after moving the axes
        new_axes_out = tuple(np.arange(len(out_axes_kept)))
        # New indices of the kept input dimensions after moving the axes
        new_axes_in = tuple(np.arange(len(in_axes_kept)) + len(new_axes_out))

        # Create the view by moving the axes
        conn = np.moveaxis(weights,
                           out_axes_kept + in_axes_kept,
                           new_axes_out + new_axes_in)
        #
        ###

        # For each source-target dimension pair, set connections to 1 for
        # every pair of neurons along that dimension, as well as to all
        # neurons in all remaining dimensions
        if smaller_num_dims == 1:
            for a in range(np.size(conn, axis=0)):
                conn[a, a, ...] = 1
        elif smaller_num_dims == 2:
            for a in range(np.size(conn, axis=0)):
                for b in range(np.size(conn, axis=1)):
                    conn[a, b, a, b, ...] = 1
        elif smaller_num_dims == 3:
            for a in range(np.size(conn, axis=0)):
                for b in range(np.size(conn, axis=1)):
                    for c in range(np.size(conn, axis=2)):
                        conn[a, b, c, a, b, c, ...] = 1
        else:
            raise NotImplementedError("projection is not implemented for "
                                      "dimensionality > 3")

        # Flatten the source and target dimensions of the connectivity
        # matrix to get a two-dimensional dense connectivity matrix
        weights = weights.reshape((num_neurons_out, num_neurons_in))

    return weights


class Convolution(AbstractOperation):
    """
    Creates connectivity that resembles a convolution with a kernel.
    Perhaps contrary to other implementations of the convolution, this
    operation always leaves the shape of the input intact. That is, a
    Convolution operation applied, for instance, to the output of a
    population of neurons of shape (42, 42) will also yield an output of
    shape (42, 42).

    Parameters
    ----------
    kernel : Kernel
        kernel of weights that the input will be convolved with; must be of the
        same dimensionality as the input
    border_types : BorderType or list(BorderType)
        determines how the Convolution operation treats borders; valid values
        are (1) PADDED, in which case the borders will be padded with a value
        that can be specified in the Kernel or (2) CIRCULAR, in which case
        the values from the other side of the input will be used as 'padding'
        (this is sometimes also called "wrapped")

    """
    def __init__(
        self,
        kernel: ty.Union[Kernel, np.ndarray],
        border_types: ty.Optional[ty.Union[BorderType,
                                           ty.List[BorderType]]]
            = BorderType.PADDED
    ) -> None:
        super().__init__(KeepShapeHandler())

        self._kernel = self._validate_kernel(kernel)
        self._border_types = self._validate_border_types(border_types)

    @property
    def kernel(self) -> Kernel:
        """Returns the kernel"""
        return self._kernel

    @property
    def border_types(self) -> ty.List[BorderType]:
        """Returns the list of border types"""
        return self._border_types

    @staticmethod
    def _validate_kernel(
        kernel: ty.Union[Kernel, np.ndarray]
    ) -> Kernel:
        """Validate the <kernel> argument"""
        if isinstance(kernel, np.ndarray):
            kernel = Kernel(weights=kernel)

        return kernel

    @staticmethod
    def _validate_border_types(
        border_types: ty.Union[BorderType, ty.List[BorderType]]
    ) -> ty.List[BorderType]:
        """Validates the <border_types> argument"""

        if isinstance(border_types, BorderType):
            border_types = [border_types]

        if not isinstance(border_types, list):
            raise TypeError("<border_types> must be of type BorderType or"
                            "list(BorderType)")

        for bt in border_types:
            BorderType.validate(bt)

        return border_types

    def _validate_args_with_input_shape(self,
                                        input_shape: ty.Tuple[int, ...]
                                        ) -> None:
        # treating 0D cases like 1D cases here
        input_dim = len(input_shape)

        if len(self._border_types) == 1:
            self._border_types *= input_dim
        if len(self._border_types) != input_dim:
            raise ValueError("number of entries in <border_type> does not"
                             "match dimensionality of population")

    def _compute_weights(self) -> np.ndarray:

        # Input shape equals output shape
        shape = self.input_shape
        # Do not use num_dims() here to treat 0D like 1D
        _num_dims = len(shape)
        _num_neurons = num_neurons(shape)

        # Generate a dense connectivity matrix
        connectivity_matrix = np.zeros((_num_neurons, _num_neurons))

        # Copy the weights of the kernel
        kernel_weights = np.copy(self.kernel.weights)

        for i in range(_num_dims):
            # Compute the size difference between the population and the
            # kernel in the current dimension
            size_diff = shape[i] - np.size(kernel_weights, axis=i)

            if size_diff != 0:
                pad_width = np.zeros((_num_dims, 2), dtype=int)
                pad_width[i, :] = int(np.floor(np.abs(size_diff) / 2.0))
                # If the padding cannot be distributed evenly...
                if is_odd(size_diff):
                    if is_odd(np.size(kernel_weights, axis=i)):
                        # ...add one in front if the kernel size is odd...
                        pad_width[i, 0] += 1
                    else:
                        # ...or add one in the back if the kernel size
                        # is even
                        pad_width[i, 1] += 1

                if size_diff > 0:
                    # Pad the kernel with its padding value
                    kernel_weights = \
                        np.pad(kernel_weights,
                               pad_width=pad_width,
                               constant_values=self.kernel.padding_value)
                elif size_diff < 0 \
                        and self.border_types[i] == BorderType.CIRCULAR:
                    delete_front = pad_width[i, 1]
                    delete_back = pad_width[i, 0]
                    kernel_weights = np.delete(kernel_weights,
                                               range(delete_front),
                                               axis=i)
                    kernel_weights = np.delete(kernel_weights,
                                               range(-delete_back, 0),
                                               axis=i)

        # Compute the center of the kernel
        kernel_center = np.floor(np.array(kernel_weights.shape) / 2.0)

        # Iterate over the shape of the input population
        for index, _ in np.ndenumerate(np.zeros(shape)):
            # Compute how much the kernel must be shifted to bring its
            # center to the correct position
            shift = kernel_center.astype(int) - np.array(index,
                                                         dtype=int)

            conn_weights = kernel_weights

            # Shift the weights depending on the border method
            for i in range(_num_dims):
                if self.border_types[i] == BorderType.CIRCULAR:
                    conn_weights = np.roll(conn_weights, -shift[i], axis=i)
                elif self.border_types[i] == BorderType.PADDED:
                    conn_weights = \
                        self._shift_fill(conn_weights,
                                         -shift[i],
                                         axis=i,
                                         fill_value=self.kernel.padding_value)

                    # If the connection weight matrix is too large for the
                    # population...
                    size_diff = shape[i] - np.size(conn_weights, axis=i)
                    if size_diff < 0:
                        # ...delete the overflowing elements
                        conn_weights = np.delete(conn_weights,
                                                 range(-np.abs(size_diff), 0),
                                                 axis=i)

            # Flatten kernel matrix
            if _num_dims > 1:
                conn_weights = np.ravel(conn_weights)

            # Fill the connectivity matrix
            flat_index = np.ravel_multi_index(index, shape)
            connectivity_matrix[flat_index, :] = conn_weights

        return connectivity_matrix

    @staticmethod
    def _shift_fill(array: np.ndarray,
                    shift: int,
                    axis: int = 0,
                    fill_value: float = 0) -> np.ndarray:
        """
        Shift an array along a given axis, filling the empty elements.

        Parameters
        ----------
        array : numpy.ndarray
            the array to be shifted
        shift : int
            number of elements to shift
        axis : int
            axis along which the array is shifted
        fill_value: float
            value that will fill up empty elements in the shifted array

        Returns
        -------
        shifted array : numpy.ndarray

        """
        if shift != 0:
            if axis > array.ndim - 1:
                raise IndexError(f"axis {axis} does not exist for array of "
                                 f"shape {array.shape}")

            array = np.swapaxes(array, 0, axis)
            shifted_array = np.empty_like(array)

            if shift < 0:
                shifted_array[shift:, ...] = fill_value
                shifted_array[:shift, ...] = array[-shift:, ...]
            elif shift > 0:
                shifted_array[:shift, ...] = fill_value
                shifted_array[shift:, ...] = array[:-shift, ...]

            shifted_array = np.swapaxes(shifted_array, axis, 0)

            return shifted_array
        else:
            return array


class ReduceDiagonal(AbstractOperation):
    """
    Creates connectivity that projects the output of a source population
    along its diagonal. For instance, if the source population is a grid of
    neurons of shape (40, 40), the operation will project (sum) the output of
    that two-dimensional population along its diagonal, yielding a
    one-dimensional output of shape (79,). The size is the length of the
    diagonal that the operation does not sum over, in this case
    79 = 40 * 2 - 1.
    """
    def __init__(self) -> None:
        super().__init__(ReduceDiagonalHandler())

    def _compute_weights(self) -> np.ndarray:
        weights = np.zeros(self.output_shape + self.input_shape,
                           dtype=np.int32)

        # Extract the original shape of the populations that are input to the
        # higher-dimensional population, which is the source of input here.
        # This assumes that self.input_shape = shape + shape.
        _num_dims = num_dims(self.input_shape)
        shape = self.input_shape[0:int(_num_dims/2)]

        shape_array = np.array(shape)
        shape_doubled = tuple(shape_array * 2)

        # Iterate over all positions within ('shape' * 2).
        # 'x' will be a tuple of the dimensionality of 'shape'
        for x in np.ndindex(*shape_doubled):
            x = np.array(x)
            # Iterate over all positions within 'shape'.
            # 'p' will be a tuple of the dimensionality of 'shape'
            for p in np.ndindex(*shape):
                p = np.array(p)

                # Set weights[x, x-p, p] = 1, where 0 <= x-p < shape
                d = x - p
                if np.all(0 <= d) and np.all(d < shape_array):
                    idx = tuple(np.concatenate([x, d, p]))
                    weights[idx] = 1

        # Reshape weights matrix to 2D;
        # shape: (number of output neurons, number of input neurons)
        return weights.reshape((np.prod(self.output_shape),) +
                               (num_neurons(self.input_shape),))

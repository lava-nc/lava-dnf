# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod
import typing as ty
import numpy as np

from lava.lib.dnf.utils.convenience import num_neurons
from lava.lib.dnf.operations.shape_handlers import AbstractShapeHandler,\
    KeepShapeHandler, ReduceDimsHandler, ExpandDimsHandler, ReorderHandler
from lava.lib.dnf.operations.enums import ReduceMethod

from lava.lib.dnf.utils.convenience import num_dims


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
    def __init__(self, shape_handler: AbstractShapeHandler):
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
        # assert that the input and output shape is configured
        self._shape_handler.assert_configured()

        # compute and return connectivity weight matrix
        return self._compute_weights()

    def configure(self,
                  input_shape: ty.Tuple[int, ...]):
        """
        Configures an operation by setting its input and output shape.

        Parameters
        ----------
        input_shape : tuple(int)
            input shape of the operation

        """
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
                 reduce_method: ty.Optional[ReduceMethod] = ReduceMethod.SUM):
        super().__init__(ReduceDimsHandler(reduce_dims))
        ReduceMethod.validate(reduce_method)
        self.reduce_method = reduce_method

    def _compute_weights(self) -> np.ndarray:
        # indices of the input dimensions in the weight matrix that will
        # not be removed; these will be come after the axes of the output
        # dimensions defined above
        in_axes_all = np.arange(num_dims(self.input_shape))
        sh = ty.cast(ReduceDimsHandler, self._shape_handler)
        in_axes_kept = tuple(np.delete(in_axes_all, sh.reduce_dims))

        # generate the weight matrix
        weights = _project_dims(self.input_shape,
                                self.output_shape,
                                in_axes_kept=in_axes_kept)

        if self.reduce_method == ReduceMethod.MEAN:
            # set the weights such that they compute the mean
            weights = weights / num_neurons(self.input_shape)

        return weights


class ExpandDims(AbstractOperation):
    """
    Operation that expands the dimensionality of the input by projecting
    the dimensions of the input to the newly added dimensions.

    """
    def __init__(self,
                 new_dims_shape: ty.Union[int, ty.Tuple[int, ...]]):
        super().__init__(ExpandDimsHandler(new_dims_shape))

    def _compute_weights(self) -> np.ndarray:
        # indices of the output dimensions in the weight matrix that will
        # be kept from the input
        out_axes_kept = tuple(np.arange(num_dims(self.input_shape)))

        # generate the weight matrix
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

    """
    def __init__(self, order: ty.Tuple[int, ...]):
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
        # if the target is a 0D population, the connectivity is from
        # all neurons in the source population to that one neuron
        weights = np.ones((num_neurons_out, num_neurons_in))
    else:
        # create a dense connectivity matrix, where dimensions of the
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

        # new indices of the kept output dimensions after moving the axes
        new_axes_out = tuple(np.arange(len(out_axes_kept)))
        # new indices of the kept input dimensions after moving the axes
        new_axes_in = tuple(np.arange(len(in_axes_kept)) + len(new_axes_out))

        # create the view by moving the axes
        conn = np.moveaxis(weights,
                           out_axes_kept + in_axes_kept,
                           new_axes_out + new_axes_in)
        #
        ###

        # for each source-target dimension pair, set connections to 1 for
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

        # flatten the source and target dimensions of the connectivity
        # matrix to get a two-dimensional dense connectivity matrix
        weights = weights.reshape((num_neurons_out, num_neurons_in))

    return weights

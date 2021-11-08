# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, ABCMeta, abstractmethod
import typing as ty
import numpy as np
from enum import Enum, unique, auto

from lava.lib.dnf.utils.convenience import num_neurons
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError

from lava.lib.dnf.utils.convenience import num_dims


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
        self._validate_input_shape()
        self._validate_reduce_dims()
        self._output_shape = tuple(np.delete(np.asarray(input_shape),
                                             self.reduce_dims))
        if self._output_shape == ():
            self._output_shape = (1,)

    def _validate_input_shape(self):
        if num_dims(self._input_shape) == 0:
            raise MisconfiguredOpError("ReduceDims operation is "
                                       "configured with an input shape that "
                                       "is already zero-dimensional")

    def _validate_reduce_dims(self):
        if len(self.reduce_dims) == 0:
            raise ValueError("<reduce_dims> may not be empty")

        if len(self.reduce_dims) > len(self._input_shape):
            raise ValueError(f"given <reduce_dims> {self.reduce_dims} has "
                             f"more entries than the shape of the input "
                             f"{self._input_shape}")

        for idx in self.reduce_dims:
            # compute the positive index irrespective of the sign of 'idx'
            idx_pos = len(self._input_shape) + idx if idx < 0 else idx
            # make sure the positive index is not out of bounds
            if 0 < idx_pos >= len(self._input_shape):
                raise IndexError(f"<reduce_dims> value {idx} is out of bounds "
                                 f"for array of size {len(self._input_shape)}")


class AbstractExpandDimsOperation(AbstractComputedShapeOperation,
                                  metaclass=ABCMeta):
    """Abstract Operation that expands the dimensionality of the input by
    concatenating additional dimensions with the given shape
    <new_dims_shape>.

    Parameters
    ----------
    new_dims_shape : int or tuple(int)
        shape of the added dimensions; they will be concatenated to the shape
        of the input, for instance an input shape (2,) and
        new_dims_shape=(6, 8) will produce an output shape (2, 6, 8)
    """
    def __init__(self,
                 new_dims_shape: ty.Union[int, ty.Tuple[int, ...]]):
        super().__init__()
        if isinstance(new_dims_shape, int):
            new_dims_shape = (new_dims_shape,)
        self.new_dims_shape = new_dims_shape

    def _compute_output_shape(self, input_shape: ty.Tuple[int, ...]):
        self._validate_new_dims_shape()
        if num_dims(input_shape) == 0:
            self._output_shape = self.new_dims_shape
        else:
            self._output_shape = input_shape + self.new_dims_shape

        if len(self._output_shape) > 3:
            raise NotImplementedError("ExpandDims operation is configured to "
                                      "produce an output shape with "
                                      "dimensionality larger than 3; higher "
                                      "dimensionality is currently not "
                                      "supported")

    def _validate_new_dims_shape(self):
        """Validate the <new_dims_shape> argument"""
        if len(self.new_dims_shape) == 0:
            raise ValueError("<new_dims_shape> may not be empty")

        if any(s < 1 for s in self.new_dims_shape):
            raise ValueError("values in <new_dims_shape> may not be smaller "
                             "than 1")


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


@unique
class ReduceMethod(Enum):
    """Enum for reduce methods of ReduceDims Operation"""
    SUM = auto()  # ReduceDims will sum all synaptic weights of collapsed dim
    MEAN = auto()  # ReduceDims will compute mean of weights of collapsed dim

    @classmethod
    def validate(cls, reduce_method):
        """Validate <reduce_op> against all valid enum values"""
        if not isinstance(reduce_method, ReduceMethod):
            raise TypeError("reduce_method must be of value ReduceMethod")


class ReduceDims(AbstractReduceDimsOperation):
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
                 reduce_method: ReduceMethod):
        super().__init__(reduce_dims)
        ReduceMethod.validate(reduce_method)
        self.reduce_method = reduce_method

    def _compute_weights(self) -> np.ndarray:
        num_dims_in = num_dims(self._input_shape)
        num_dims_out = num_dims(self._output_shape)

        # indices of the output dimensions in the weight matrix that will be
        # kept from the input, which in this case is all of them;
        # these will be just the first <num_dims_out> number of axes in the
        # weight matrix
        orig_axes_out = np.arange(num_dims_out)
        # indices of the input dimensions in the weight matrix that will
        # not be removed; these will be come after the axes of the output
        # dimensions defined above
        orig_axes_in = np.delete(np.arange(num_dims_in),
                                 self.reduce_dims) + num_dims_out

        # generate the weight matrix
        weights = _project_dims(self._input_shape,
                                self._output_shape,
                                tuple(orig_axes_out),
                                tuple(orig_axes_in))

        if self.reduce_method == ReduceMethod.MEAN:
            # set the weights such that they compute the mean
            weights = weights / num_neurons(self._input_shape)

        return weights


class ExpandDims(AbstractExpandDimsOperation):
    """
    Operation that expands the dimensionality of the input by projecting
    the dimensions of the input to the newly added dimensions.

    """
    def __init__(self,
                 new_dims_shape: ty.Union[int, ty.Tuple[int, ...]]):
        super().__init__(new_dims_shape)

    def _compute_weights(self) -> np.ndarray:
        # indices of the output dimensions in the weight matrix that were
        # already present in the input
        orig_axes_out = np.arange(num_dims(self._input_shape))
        # indices of the input dimensions in the weight matrix that will
        # be kept for the output, which in this case is all of them;
        # these will come after the axes of the output
        # dimensions defined above
        orig_axes_in = orig_axes_out + num_dims(self._output_shape)

        # generate the weight matrix
        weights = _project_dims(self._input_shape,
                                self._output_shape,
                                tuple(orig_axes_out),
                                tuple(orig_axes_in))

        return weights


def _project_dims(input_shape: ty.Tuple[int, ...],
                  output_shape: ty.Tuple[int, ...],
                  orig_axes_out: ty.Tuple[int, ...],
                  orig_axes_in: ty.Tuple[int, ...]) -> np.ndarray:
    """Projection function that is used both by the ReduceDims and ExpandDims
    Operation

    Parameters
    ----------
    input_shape : tuple(int)
        input shape of the operation
    output_shape : tuple(int)
        output shape of the operation
    orig_axes_out : tuple(int)
        indices of the output dimensions that will carry over from the input;
        these indices are with respect to the overall weight matrix (before
        flattening) that is created in this function
    orig_axes_in : tuple(int)
        indices of the input dimensions that will carry over;
        these indices are with respect to the overall weight matrix (before
        flattening) that is created in this function

    Returns
    -------
    connectivity weight matrix : numpy.ndarray
    """
    num_neurons_in = num_neurons(input_shape)
    num_neurons_out = num_neurons(output_shape)
    smaller_num_dims = min(num_dims(input_shape), num_dims(output_shape))

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
        # target dimensions, followed by all source dimensions, followed by
        # all remaining dimensions.

        # new indices of the target dimensions after moving the axes
        new_axes_out = np.arange(len(orig_axes_out))
        # new indices of the source dimensions after moving the axes
        new_axes_in = np.arange(len(orig_axes_in)) + len(new_axes_out)

        # create the view by moving the axes
        conn = np.moveaxis(weights,
                           orig_axes_out + orig_axes_in,
                           tuple(new_axes_out) + tuple(new_axes_in))
        #
        ###

        # for each source-target dimension pair, set connections to 1 for
        # every pair of neurons along that dimension, as well as to all
        # neurons in all remaining dimensions
        if smaller_num_dims == 1:
            for a in range(np.size(conn, axis=0)):
                conn[a, a, ...] = 1
        if smaller_num_dims == 2:
            for a in range(np.size(conn, axis=0)):
                for b in range(np.size(conn, axis=1)):
                    conn[a, b, a, b, ...] = 1

        # flatten the source and target dimensions of the connectivity
        # matrix to get a two-dimensional dense connectivity matrix
        weights = weights.reshape((num_neurons_out, num_neurons_in))

    return weights

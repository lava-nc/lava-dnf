# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty
import numpy as np

from lava.lib.dnf.operations.operations import AbstractOperation, \
    AbstractComputedShapeOperation, AbstractSpecifiedShapeOperation, Weights,\
    AbstractKeepShapeOperation, AbstractReduceDimsOperation, \
    AbstractReshapeOperation
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError

from lava.lib.dnf.utils.convenience import num_neurons


class MockOperation(AbstractOperation):
    """Generic mock Operation"""
    def _compute_weights(self) -> np.ndarray:
        return np.ones((1, 1), dtype=np.int32)

    def configure(self, input_shape: ty.Tuple[int, ...]):
        self._input_shape = input_shape
        self._output_shape = input_shape


class MockComputedShapeOperation(AbstractComputedShapeOperation):
    """Mock Operation whose output shape is computed from its input shape"""
    def _compute_weights(self) -> np.ndarray:
        return np.ones((1, 1), dtype=np.int32)

    def _compute_output_shape(self, input_shape: ty.Tuple[int, ...]):
        self._output_shape = input_shape


class MockSpecifiedShapeOperation(AbstractSpecifiedShapeOperation):
    """Mock Operation whose output shape is specified by the user"""
    def __init__(self, output_shape):
        super().__init__(output_shape)

    def _compute_weights(self) -> np.ndarray:
        return np.ones((1, 1), dtype=np.int32)

    def _validate_output_shape(self):
        # some arbitrary validation criterion to test validation
        if self._input_shape != self._output_shape:
            raise MisconfiguredOpError("testing validation")


class MockKeepShapeOp(AbstractKeepShapeOperation):
    """Mock Operation that does not change its shape"""
    def _compute_weights(self):
        return np.ones((1, 1), dtype=np.int32)


class MockReduceDims(AbstractReduceDimsOperation):
    """Mock Operation that reduces the dimensionality of the incoming matrix"""
    def _compute_weights(self):
        return np.ones((1, 1), dtype=np.int32)


class MockReshapeOp(AbstractReshapeOperation):
    """Mock Operation that reshapes the incoming matrix"""
    def _compute_weights(self):
        return np.ones((1, 1), dtype=np.int32)


class TestAbstractOperation(unittest.TestCase):
    def test_computing_conn_without_prior_configuration_raises_error(self):
        """Tests whether an error is raised when compute_weights() is called
        before an operation has been configured."""
        op = MockOperation()
        with self.assertRaises(AssertionError):
            op.compute_weights()

    def test_output_shape_getter(self):
        """Tests whether the output shape property works."""
        op = MockOperation()
        shape = (2, 4)
        op._output_shape = shape
        self.assertEqual(op.output_shape, shape)

    def test_computing_conn_with_prior_configuration_works(self):
        """Tests whether compute_weights() works and can be called once
        configuration is complete."""
        op = MockOperation()
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 1), dtype=np.int32)

        self.assertEqual(computed_weights, expected_weights)


class TestAbstractComputedShapeOperation(unittest.TestCase):
    def test_configure_sets_input_shape(self):
        """Tests whether the configure() method sets the input shape."""
        input_shape = (2, 4)
        op = MockComputedShapeOperation()
        op.configure(input_shape=input_shape)
        self.assertEqual(op._input_shape, input_shape)

    def test_configure(self):
        """Tests whether the configure() method executes
        _compute_output_shape()."""
        op = MockComputedShapeOperation()
        input_shape = (2, 4)
        op.configure(input_shape=input_shape)
        self.assertEqual(op._input_shape, input_shape)
        self.assertEqual(op.output_shape, input_shape)


class TestAbstractSpecifiedShapeOperation(unittest.TestCase):
    def test_specifying_output_shape(self):
        """Tests whether the output shape can be specified in the __init__
        method."""
        output_shape = (2, 4)
        op = MockSpecifiedShapeOperation(output_shape)
        self.assertEqual(op.output_shape, output_shape)

    def test_configure_sets_input_shape(self):
        """Tests whether the configure() method sets the input shape."""
        input_shape = (2, 4)
        op = MockSpecifiedShapeOperation((2, 4))
        op.configure(input_shape=input_shape)
        self.assertEqual(op._input_shape, input_shape)

    def test_configure_validates_output_shape(self):
        """Tests whether the configure() method validates the output shape."""
        input_shape = (2, 4)
        op = MockSpecifiedShapeOperation((5, 3))
        with self.assertRaises(MisconfiguredOpError):
            op.configure(input_shape)


class TestKeepShapeOperation(unittest.TestCase):
    def test_compute_output_shape(self):
        """Tests whether the output shape is set correctly."""
        op = MockKeepShapeOp()
        input_shape = (2, 4)
        op.configure(input_shape=input_shape)
        self.assertEqual(op.output_shape, input_shape)


class TestReduceDimsOperation(unittest.TestCase):
    def test_compute_output_shape_remove_one_dim(self):
        """Tests whether the output shape is set correctly when a single
        dimension is removed."""
        op = MockReduceDims(reduce_dims=1)
        op.configure(input_shape=(2, 4))
        self.assertEqual(op.output_shape, (2,))

    def test_compute_output_shape_negative_index(self):
        """Tests whether the output shape is set correctly when a single
        dimension is removed, using a negative index."""
        op = MockReduceDims(reduce_dims=-1)
        op.configure(input_shape=(2, 4))
        self.assertEqual(op.output_shape, (2,))

    def test_compute_output_shape_remove_multiple_dims(self):
        """Tests whether the output shape is set correctly when multiple
        dimensions are removed."""
        op = MockReduceDims(reduce_dims=(0, -1))
        op.configure(input_shape=(2, 3, 4, 5))
        self.assertEqual(op.output_shape, (3, 4))

    def test_reduce_dims_with_out_of_bounds_index_raises_error(self):
        """Tests whether an error is raised when <reduce_dims> contains an
        index that is out of bounds for the input shape of the operation."""
        with self.assertRaises(IndexError):
            op = MockReduceDims(reduce_dims=2)
            op.configure(input_shape=(2, 4))
            op.compute_weights()

    def test_reduce_dims_with_negative_out_of_bounds_index_raises_error(self):
        """Tests whether an error is raised when <reduce_dims> contains a
        negative index that is out of bounds for the input shape of the
        operation."""
        with self.assertRaises(IndexError):
            op = MockReduceDims(reduce_dims=-3)
            op.configure(input_shape=(2, 4))
            op.compute_weights()

    def test_empty_reduce_dims_raises_error(self):
        """Tests whether an error is raised when <reduce_dims> is an empty
        tuple."""
        with self.assertRaises(ValueError):
            op = MockReduceDims(reduce_dims=())
            op.configure(input_shape=(2, 4))
            op.compute_weights()


class TestReshapeOperation(unittest.TestCase):
    def test_compute_output_shape(self):
        """Tests whether the output shape is set correctly."""
        output_shape = (8,)
        op = MockReshapeOp(output_shape=output_shape)
        op.configure(input_shape=(2, 4))
        self.assertEqual(op.output_shape, output_shape)

    def test_compute_output_shape_with_incorrect_num_neurons_raises_error(self):
        """Tests whether an error is raised when the number of neurons in
        the input and output shape does not match."""
        with self.assertRaises(MisconfiguredOpError):
            output_shape = (9,)
            op = MockReshapeOp(output_shape=output_shape)
            op.configure(input_shape=(2, 4))
            op.compute_weights()


class TestWeights(unittest.TestCase):
    def test_init(self):
        """Tests whether a Weights operation can be instantiated."""
        weights_op = Weights(weight=5.0)
        self.assertIsInstance(weights_op, Weights)

    def test_output_shape_is_set_correctly(self):
        """Tests whether the output shape is set to be equal to the input
        shape."""
        weights_op = Weights(weight=5.0)
        input_shape = (5, 3)
        weights_op.configure(input_shape)
        self.assertEqual(input_shape, weights_op.output_shape)

    def test_compute_weights(self):
        """Tests whether computing weights produces the expected result for
        shapes of different dimensionality."""
        for shape in [(1,), (5,), (5, 3), (5, 3, 2)]:
            w = 5.0
            weights_op = Weights(weight=w)
            weights_op.configure(shape)

            computed_weights = weights_op.compute_weights()
            expected_weights = np.eye(num_neurons(shape),
                                      num_neurons(shape),
                                      dtype=np.int32) * w

            self.assertTrue(np.array_equal(computed_weights, expected_weights))


if __name__ == '__main__':
    unittest.main()

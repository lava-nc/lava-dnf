# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty
import numpy as np

from lava.lib.dnf.operations.operations import AbstractOperation, \
    AbstractComputedShapeOperation, AbstractSpecifiedShapeOperation, \
    AbstractKeepShapeOperation, AbstractReduceDimsOperation, \
    AbstractReshapeOperation, Weights, ReduceDims, ReduceMethod
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

    def test_compute_output_shape_remove_all(self):
        """Tests whether the output shape is set correctly when all
        dimensions are removed."""
        op = MockReduceDims(reduce_dims=(0, 1, 2))
        op.configure(input_shape=(2, 3, 4))
        self.assertEqual(op.output_shape, (1,))

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

    def test_reduce_dims_with_too_many_entries_raises_error(self):
        """Tests whether an error is raised when <reduce_dims> has more
        elements than the dimensionality of the input."""
        with self.assertRaises(ValueError):
            op = MockReduceDims(reduce_dims=(0, 0))
            op.configure(input_shape=(4,))
            op.compute_weights()

    def test_zero_dimensional_input_shape_raises_error(self):
        """Tests whether an error is raised when the input shape is already
        zero-dimensional."""
        with self.assertRaises(MisconfiguredOpError):
            op = MockReduceDims(reduce_dims=0)
            op.configure(input_shape=(1,))


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

    def test_weight_is_set_correctly(self):
        """Tests whether the weight is set correctly.
        shape."""
        weight = 5.0
        op = Weights(weight=weight)
        self.assertEqual(op.weight, weight)

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


class TestReduceDims(unittest.TestCase):
    def test_init(self):
        """Tests whether a ReduceDims operation can be instantiated."""
        reduce_method = ReduceMethod.SUM
        op = ReduceDims(reduce_dims=0,
                        reduce_method=reduce_method)

        self.assertIsInstance(op, ReduceDims)
        self.assertEqual(op.reduce_method, reduce_method)

    def test_compute_weights_2d_to_0d_sum(self):
        """Tests reducing dimensionality from 2D to 0D using SUM."""
        op = ReduceDims(reduce_dims=(0, 1),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 9))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_2d_to_0d_mean(self):
        """Tests reducing dimensionality from 2D to 0D using MEAN."""
        op = ReduceDims(reduce_dims=(0, 1),
                        reduce_method=ReduceMethod.MEAN)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 9)) / 9.0

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_reduce_method_mean(self):
        """Tests whether MEAN produces the same results as SUM divided by the
        number of elements in the reduced dimension."""
        reduce_dims = (1,)
        input_shape = (3, 3)
        op_sum = ReduceDims(reduce_dims=reduce_dims,
                            reduce_method=ReduceMethod.SUM)
        op_sum.configure(input_shape=input_shape)
        computed_weights_sum = op_sum.compute_weights()

        op_mean = ReduceDims(reduce_dims=reduce_dims,
                             reduce_method=ReduceMethod.MEAN)
        op_mean.configure(input_shape=input_shape)
        computed_weights_mean = op_mean.compute_weights()

        self.assertTrue(np.array_equal(computed_weights_mean,
                                       computed_weights_sum / 9.0))

    def test_compute_weights_2d_to_1d_reduce_axis_0_sum(self):
        """Tests reducing dimension 0 from 2D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(0,),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 1, 0, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_2d_to_1d_axis_reduce_axis_1_sum(self):
        """Tests reducing dimension 1 from 2D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(1,),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 1, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_1d_axis_keep_axis_0_sum(self):
        """Tests reducing dimensions 1 and 2 from 3D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(1, 2),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 1, 1, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_1d_axis_keep_axis_1_sum(self):
        """Tests reducing dimensions 0 and 2 from 3D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(0, 2),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 1, 0, 0, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 0, 0, 1, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_1d_axis_keep_axis_2_sum(self):
        """Tests reducing dimensions 0 and 1 from 3D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(0, 1),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                                     [0, 1, 0, 1, 0, 1, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_2d_axis_reduce_axis_0_sum(self):
        """Tests reducing dimension 0 from 3D to 2D using SUM."""
        op = ReduceDims(reduce_dims=(0,),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_2d_axis_reduce_axis_1_sum(self):
        """Tests reducing dimension 1 from 3D to 2D using SUM."""
        op = ReduceDims(reduce_dims=(1,),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_2d_axis_reduce_axis_2_sum(self):
        """Tests reducing dimension 2 from 3D to 2D using SUM."""
        op = ReduceDims(reduce_dims=(2,),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))


class TestReduceMethod(unittest.TestCase):
    def test_validate_sum(self):
        """Tests whether SUM is a valid type of the ReduceMethod enum."""
        ReduceMethod.validate(ReduceMethod.SUM)

    def test_validate_mean(self):
        """Tests whether MEAN is a valid type of the ReduceMethod enum."""
        ReduceMethod.validate(ReduceMethod.MEAN)

    def test_invalid_type_raises_type_error(self):
        """Tests whether int is an invalid type of the ReduceMethod enum."""
        with self.assertRaises(TypeError):
            ReduceMethod.validate(int)

    def test_invalid_value_raises_value_error(self):
        """Tests whether FOO is an invalid value of the ReduceMethod enum."""
        with self.assertRaises(AttributeError):
            _ = ReduceMethod.FOO


if __name__ == '__main__':
    unittest.main()

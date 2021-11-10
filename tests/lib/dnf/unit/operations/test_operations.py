# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.dnf.operations.operations import AbstractOperation, \
    Weights, ReduceDims, ReduceMethod, ExpandDims, Reorder
from lava.lib.dnf.operations.shape_handlers import KeepShapeHandler

from lava.lib.dnf.utils.convenience import num_neurons


class MockOperation(AbstractOperation):
    """Generic mock Operation"""
    def __init__(self):
        super().__init__(shape_handler=KeepShapeHandler())

    def _compute_weights(self) -> np.ndarray:
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
        op._shape_handler._output_shape = shape
        self.assertEqual(op.output_shape, shape)

    def test_input_shape_getter(self):
        """Tests whether the input shape property works."""
        op = MockOperation()
        shape = (2, 4)
        op._shape_handler._input_shape = shape
        self.assertEqual(op.input_shape, shape)

    def test_computing_conn_with_prior_configuration_works(self):
        """Tests whether compute_weights() works and can be called once
        configuration is complete."""
        op = MockOperation()
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 1), dtype=np.int32)

        self.assertEqual(computed_weights, expected_weights)

    def test_configure_sets_input_and_output_shape(self):
        """Tests whether the configure() method sets the input and
        output shape."""
        input_shape = (2, 4)
        op = MockOperation()
        op.configure(input_shape=input_shape)
        self.assertEqual(op.input_shape, input_shape)
        self.assertEqual(op.output_shape, input_shape)


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


class TestExpandDims(unittest.TestCase):
    def test_init(self):
        """Tests whether an ExpandDims operation can be instantiated."""
        op = ExpandDims(new_dims_shape=(5,))
        self.assertIsInstance(op, ExpandDims)

    def test_compute_weights_0d_to_1d(self):
        """Tests expanding dimensionality from 0D to 1D."""
        op = ExpandDims(new_dims_shape=(3,))
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((3, 1))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_0d_to_2d(self):
        """Tests expanding dimensionality from 0D to 2D."""
        op = ExpandDims(new_dims_shape=(3, 3))
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((9, 1))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_0d_to_3d(self):
        """Tests expanding dimensionality from 0D to 3D."""
        op = ExpandDims(new_dims_shape=(3, 3, 3))
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((27, 1))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_1d_to_2d(self):
        """Tests expanding dimensionality from 1D to 2D."""
        op = ExpandDims(new_dims_shape=(3,))
        op.configure(input_shape=(3,))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 0],
                                     [1, 0, 0],
                                     [1, 0, 0],

                                     [0, 1, 0],
                                     [0, 1, 0],
                                     [0, 1, 0],

                                     [0, 0, 1],
                                     [0, 0, 1],
                                     [0, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_1d_to_3d(self):
        """Tests expanding dimensionality from 1D to 3D."""
        op = ExpandDims(new_dims_shape=(2, 2))
        op.configure(input_shape=(2,))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0],
                                     [1, 0],
                                     [1, 0],
                                     [1, 0],
                                     [0, 1],
                                     [0, 1],
                                     [0, 1],
                                     [0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_2d_to_3d(self):
        """Tests expanding dimensionality from 2D to 3D."""
        op = ExpandDims(new_dims_shape=(2,))
        op.configure(input_shape=(2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1],
                                     [0, 0, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))


class TestReorder(unittest.TestCase):
    def test_init(self):
        """Tests whether a Reorder operation can be instantiated."""
        op = Reorder(order=(1, 0, 2))
        self.assertIsInstance(op, Reorder)

    def test_compute_weights_no_change_2d(self):
        """Tests 'reordering' a 2D input to the same order."""
        op = Reorder(order=(0, 1))
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.eye(9)

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_reordered_2d(self):
        """Tests reordering a 2D input by switching the dimensions."""
        op = Reorder(order=(1, 0))
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_reordered_3d(self):
        """Tests reordering a 3D input by switching the dimensions in all
        possible combinations."""
        orders = [(0, 1, 2),
                  (0, 2, 1),
                  (1, 0, 2),
                  (1, 2, 0),
                  (2, 0, 1),
                  (2, 1, 0)]

        expected_weights = [
            np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])]

        for order, expected in zip(orders, expected_weights):
            op = Reorder(order=order)
            op.configure(input_shape=(2, 2, 2))
            computed = op.compute_weights()

            self.assertTrue(np.array_equal(computed, expected))


if __name__ == '__main__':
    unittest.main()

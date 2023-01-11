# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty
import itertools as it

from lava.lib.dnf.operations.operations import (
    AbstractOperation,
    Weights,
    ReduceDims,
    ReduceMethod,
    ExpandDims,
    ReorderDims,
    Convolution,
    ReduceAlongDiagonal,
    ExpandAlongDiagonal,
    Flip)
from lava.lib.dnf.operations.enums import BorderType
from lava.lib.dnf.operations.shape_handlers import KeepShapeShapeHandler
from lava.lib.dnf.kernels.kernels import Kernel
from lava.lib.dnf.utils.convenience import num_neurons


class MockOperation(AbstractOperation):
    """Generic mock Operation"""
    def __init__(self) -> None:
        super().__init__(shape_handler=KeepShapeShapeHandler())

    def _compute_weights(self) -> np.ndarray:
        return np.ones((1, 1), dtype=np.int32)


class TestAbstractOperation(unittest.TestCase):
    def test_computing_conn_without_prior_configuration_raises_error(self)\
            -> None:
        """Tests whether an error is raised when compute_weights() is called
        before an operation has been configured."""
        op = MockOperation()
        with self.assertRaises(AssertionError):
            op.compute_weights()

    def test_output_shape_getter(self) -> None:
        """Tests whether the output shape property works."""
        op = MockOperation()
        shape = (2, 4)
        op._shape_handler._output_shape = shape
        self.assertEqual(op.output_shape, shape)

    def test_input_shape_getter(self) -> None:
        """Tests whether the input shape property works."""
        op = MockOperation()
        shape = (2, 4)
        op._shape_handler._input_shape = shape
        self.assertEqual(op.input_shape, shape)

    def test_computing_conn_with_prior_configuration_works(self) -> None:
        """Tests whether compute_weights() works and can be called once
        configuration is complete."""
        op = MockOperation()
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 1), dtype=np.int32)

        self.assertEqual(computed_weights, expected_weights)

    def test_configure_sets_input_and_output_shape(self) -> None:
        """Tests whether the configure() method sets the input and
        output shape."""
        input_shape = (2, 4)
        op = MockOperation()
        op.configure(input_shape=input_shape)
        self.assertEqual(op.input_shape, input_shape)
        self.assertEqual(op.output_shape, input_shape)


class TestWeights(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a Weights operation can be instantiated."""
        weights_op = Weights(weight=5.0)
        self.assertIsInstance(weights_op, Weights)

    def test_weight_is_set_correctly(self) -> None:
        """Tests whether the weight is set correctly.
        shape."""
        weight = 5.0
        op = Weights(weight=weight)
        self.assertEqual(op.weight, weight)

    def test_compute_weights(self) -> None:
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
    def test_init(self) -> None:
        """Tests whether a ReduceDims operation can be instantiated."""
        reduce_method = ReduceMethod.SUM
        op = ReduceDims(reduce_dims=0,
                        reduce_method=reduce_method)

        self.assertIsInstance(op, ReduceDims)
        self.assertEqual(op.reduce_method, reduce_method)

    def test_compute_weights_2d_to_0d_sum(self) -> None:
        """Tests reducing dimensionality from 2D to 0D using SUM."""
        op = ReduceDims(reduce_dims=(0, 1),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 9))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_2d_to_0d_mean(self) -> None:
        """Tests reducing dimensionality from 2D to 0D using MEAN."""
        op = ReduceDims(reduce_dims=(0, 1),
                        reduce_method=ReduceMethod.MEAN)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 9)) / 9.0

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_reduce_method_mean(self) -> None:
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

    def test_compute_weights_2d_to_1d_reduce_axis_0_sum(self) -> None:
        """Tests reducing dimension 0 from 2D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(0,),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 1, 0, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_2d_to_1d_axis_reduce_axis_1_sum(self) -> None:
        """Tests reducing dimension 1 from 2D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(1,),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 1, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_1d_axis_keep_axis_0_sum(self) -> None:
        """Tests reducing dimensions 1 and 2 from 3D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(1, 2),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 1, 1, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_1d_axis_keep_axis_1_sum(self) -> None:
        """Tests reducing dimensions 0 and 2 from 3D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(0, 2),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 1, 0, 0, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 0, 0, 1, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_1d_axis_keep_axis_2_sum(self) -> None:
        """Tests reducing dimensions 0 and 1 from 3D to 1D using SUM."""
        op = ReduceDims(reduce_dims=(0, 1),
                        reduce_method=ReduceMethod.SUM)
        op.configure(input_shape=(2, 2, 2))
        computed_weights = op.compute_weights()
        expected_weights = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                                     [0, 1, 0, 1, 0, 1, 0, 1]])

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_3d_to_2d_axis_reduce_axis_0_sum(self) -> None:
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

    def test_compute_weights_3d_to_2d_axis_reduce_axis_1_sum(self) -> None:
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

    def test_compute_weights_3d_to_2d_axis_reduce_axis_2_sum(self) -> None:
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


class TestExpandDims(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether an ExpandDims operation can be instantiated."""
        op = ExpandDims(new_dims_shape=(5,))
        self.assertIsInstance(op, ExpandDims)

    def test_compute_weights_0d_to_1d(self) -> None:
        """Tests expanding dimensionality from 0D to 1D."""
        op = ExpandDims(new_dims_shape=(3,))
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((3, 1))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_0d_to_2d(self) -> None:
        """Tests expanding dimensionality from 0D to 2D."""
        op = ExpandDims(new_dims_shape=(3, 3))
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((9, 1))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_0d_to_3d(self) -> None:
        """Tests expanding dimensionality from 0D to 3D."""
        op = ExpandDims(new_dims_shape=(3, 3, 3))
        op.configure(input_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((27, 1))

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_1d_to_2d(self) -> None:
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

    def test_compute_weights_1d_to_3d(self) -> None:
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

    def test_compute_weights_2d_to_3d(self) -> None:
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
    def test_init(self) -> None:
        """Tests whether a ReorderDims operation can be instantiated."""
        op = ReorderDims(order=(1, 0, 2))
        self.assertIsInstance(op, ReorderDims)

    def test_compute_weights_no_change_2d(self) -> None:
        """Tests 'reordering' a 2D input to the same order."""
        op = ReorderDims(order=(0, 1))
        op.configure(input_shape=(3, 3))
        computed_weights = op.compute_weights()
        expected_weights = np.eye(9)

        self.assertTrue(np.array_equal(computed_weights, expected_weights))

    def test_compute_weights_reordered_2d(self) -> None:
        """Tests reordering a 2D input by switching the dimensions."""
        op = ReorderDims(order=(1, 0))
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

    def test_compute_weights_reordered_3d(self) -> None:
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
            op = ReorderDims(order=order)
            op.configure(input_shape=(2, 2, 2))
            computed = op.compute_weights()

            self.assertTrue(np.array_equal(computed, expected))


class TestConvolution(unittest.TestCase):
    class MockKernel(Kernel):
        def __init__(self, weights: np.ndarray = None) -> None:
            if weights is None:
                weights = np.zeros((1,))
            super().__init__(weights=weights)

    def test_init(self) -> None:
        """Tests whether a Convolution operation can be instantiated while
        passing in a kernel."""
        kernel = TestConvolution.MockKernel()
        op = Convolution(kernel)
        # convolution is instantiated
        self.assertIsInstance(op, Convolution)
        # kernel is set
        self.assertEqual(op.kernel, kernel)
        # border type defaults to PADDED
        self.assertEqual(op.border_types[0], BorderType.PADDED)

    def test_kernel_of_type_numpy_ndarray(self) -> None:
        """Tests whether a <kernel> argument of type numpy.ndarray is
        converted internally into an AbstractKernel instance."""
        numpy_kernel = np.zeros((1,))
        op = Convolution(numpy_kernel)
        self.assertIsInstance(op.kernel, Kernel)

    def test_setting_valid_border_type(self) -> None:
        """Tests whether a valid border type is set correctly."""
        kernel = TestConvolution.MockKernel()
        border_type = BorderType.CIRCULAR
        op = Convolution(kernel, border_type)
        self.assertEqual(op.border_types[0], border_type)

    def test_invalid_type_of_border_type(self) -> None:
        """Checks whether a border type with an invalid type throws an
        exception."""
        with self.assertRaises(Exception) as context:
            Convolution(TestConvolution.MockKernel(),
                        border_types=["padded"])
        self.assertIsInstance(context.exception, TypeError)

    def test_invalid_border_type_list(self) -> None:
        """Checks whether a list containing an invalid border type raises
        and error."""
        border_types = [BorderType.PADDED, "circular"]
        with self.assertRaises(TypeError):
            Convolution(TestConvolution.MockKernel(),
                        border_types=border_types)

    def test_specifying_single_border_type_for_all_input_dimensions(self)\
            -> None:
        """Checks whether you can specify a single border type for all
        dimensions of the input."""
        kernel = TestConvolution.MockKernel()
        border_type_in = BorderType.CIRCULAR
        op = Convolution(kernel, border_type_in)
        input_shape = (2, 2)
        op.configure(input_shape=input_shape)

        self.assertEqual(len(op.border_types), len(input_shape))

        for border_type_op in op.border_types:
            self.assertEqual(border_type_op, border_type_in)

    def test_specifying_more_border_types_than_input_dims_raises_error(self)\
            -> None:
        """Checks whether specifying too many border types raises an
        exception."""
        kernel = TestConvolution.MockKernel()
        input_shape = (2, 2)
        border_types = [BorderType.CIRCULAR] * (len(input_shape) + 1)
        op = Convolution(kernel, border_types)

        with self.assertRaises(ValueError):
            op.configure(input_shape=input_shape)

    def _test_compute_weights(
        self,
        kernel_weights: np.ndarray,
        border_types: ty.Union[BorderType, ty.List[BorderType]],
        input_shapes: ty.List[ty.Tuple[int, ...]],
        expected_weights: ty.List[np.ndarray]
    ) -> None:
        """Helper method to test compute_weights() method"""
        kernel = TestConvolution.MockKernel(kernel_weights)

        for input_shape, expected in zip(input_shapes, expected_weights):
            with self.subTest(msg=f"input shape: {input_shape}"):
                op = Convolution(kernel, border_types=border_types)
                op.configure(input_shape)
                computed = op.compute_weights()

                self.assertTrue(np.array_equal(computed, expected))

    def test_compute_weights_0d_padded(self) -> None:
        """Tests whether the Convolution operation can be applied to 0D
        inputs with PADDED border type. It may not make sense to do this but
        it is possible."""
        self._test_compute_weights(
            kernel_weights=np.array([2]),
            border_types=BorderType.PADDED,
            input_shapes=[(1,)],
            expected_weights=[np.array([[2]])]
        )

    def test_compute_weights_0d_circular(self) -> None:
        """Tests whether the Convolution operation can be applied to 0D
        inputs with CIRCULAR border type. It may not make sense to do this but
        it is possible."""
        self._test_compute_weights(
            kernel_weights=np.array([2]),
            border_types=BorderType.CIRCULAR,
            input_shapes=[(1,)],
            expected_weights=[np.array([[2]])]
        )

    def test_connectivity_matrix_1d_odd_kernel_padded(self) -> None:
        """Tests whether computing weights works for 1D inputs with an odd sized
        kernel and PADDED border type. The input sizes cover all relevant
        cases, where the input size is smaller, equal to, and larger than the
        kernel size."""
        expected_weights = [
            np.array([[2, 3],
                      [1, 2]]),
            np.array([[2, 3, 0],
                      [1, 2, 3],
                      [0, 1, 2]]),
            np.array([[2, 3, 0, 0],
                      [1, 2, 3, 0],
                      [0, 1, 2, 3],
                      [0, 0, 1, 2]]),
            np.array([[2, 3, 0, 0, 0],
                      [1, 2, 3, 0, 0],
                      [0, 1, 2, 3, 0],
                      [0, 0, 1, 2, 3],
                      [0, 0, 0, 1, 2]])]

        self._test_compute_weights(
            kernel_weights=np.array([1, 2, 3]),
            border_types=BorderType.PADDED,
            input_shapes=[(2,), (3,), (4,), (5,)],
            expected_weights=expected_weights
        )

    def test_connectivity_matrix_1d_odd_kernel_circular(self) -> None:
        """Tests whether computing weights works for 1D inputs with an odd sized
        kernel and CIRCULAR border type."""
        expected_weights = [
            np.array([[2, 1],
                      [1, 2]]),
            np.array([[2, 3, 1],
                      [1, 2, 3],
                      [3, 1, 2]]),
            np.array([[2, 3, 0, 1],
                      [1, 2, 3, 0],
                      [0, 1, 2, 3],
                      [3, 0, 1, 2]]),
            np.array([[2, 3, 0, 0, 1],
                      [1, 2, 3, 0, 0],
                      [0, 1, 2, 3, 0],
                      [0, 0, 1, 2, 3],
                      [3, 0, 0, 1, 2]])
        ]

        self._test_compute_weights(
            kernel_weights=np.array([1, 2, 3]),
            border_types=BorderType.CIRCULAR,
            input_shapes=[(2,), (3,), (4,), (5,)],
            expected_weights=expected_weights
        )

    def test_connectivity_matrix_1d_even_kernel_padded(self) -> None:
        """Tests whether computing weights works for 1D inputs with an even
        sized kernel and PADDED border type."""
        expected_weights = [
            np.array([[3, 4],
                      [2, 3]]),
            np.array([[3, 4, 0],
                      [2, 3, 4],
                      [1, 2, 3]]),
            np.array([[3, 4, 0, 0],
                      [2, 3, 4, 0],
                      [1, 2, 3, 4],
                      [0, 1, 2, 3]]),
            np.array([[3, 4, 0, 0, 0],
                      [2, 3, 4, 0, 0],
                      [1, 2, 3, 4, 0],
                      [0, 1, 2, 3, 4],
                      [0, 0, 1, 2, 3]]),
            np.array([[3, 4, 0, 0, 0, 0],
                      [2, 3, 4, 0, 0, 0],
                      [1, 2, 3, 4, 0, 0],
                      [0, 1, 2, 3, 4, 0],
                      [0, 0, 1, 2, 3, 4],
                      [0, 0, 0, 1, 2, 3]])
        ]

        self._test_compute_weights(
            kernel_weights=np.array([1, 2, 3, 4]),
            border_types=BorderType.PADDED,
            input_shapes=[(2,), (3,), (4,), (5,), (6,)],
            expected_weights=expected_weights
        )

    def test_connectivity_matrix_1d_even_kernel_circular(self) -> None:
        """Tests whether computing weights works for 1D inputs with an even
        sized kernel and CIRCULAR border type."""
        expected_weights = [
            np.array([[3, 2],
                      [2, 3]]),
            np.array([[3, 4, 2],
                      [2, 3, 4],
                      [4, 2, 3]]),
            np.array([[3, 4, 1, 2],
                      [2, 3, 4, 1],
                      [1, 2, 3, 4],
                      [4, 1, 2, 3]]),
            np.array([[3, 4, 0, 1, 2],
                      [2, 3, 4, 0, 1],
                      [1, 2, 3, 4, 0],
                      [0, 1, 2, 3, 4],
                      [4, 0, 1, 2, 3]]),
            np.array([[3, 4, 0, 0, 1, 2],
                      [2, 3, 4, 0, 0, 1],
                      [1, 2, 3, 4, 0, 0],
                      [0, 1, 2, 3, 4, 0],
                      [0, 0, 1, 2, 3, 4],
                      [4, 0, 0, 1, 2, 3]])
        ]

        self._test_compute_weights(
            kernel_weights=np.array([1, 2, 3, 4]),
            border_types=BorderType.CIRCULAR,
            input_shapes=[(2,), (3,), (4,), (5,), (6,)],
            expected_weights=expected_weights
        )

    def test_connectivity_matrix_2d_odd_kernel_padded(self) -> None:
        """Tests whether computing weights works for 2D inputs with an odd
        sized kernel and PADDED border type."""
        expected_weights = [
            np.array([[5]]),

            np.array([[5, 6, 8, 9],
                      [4, 5, 7, 8],

                      [2, 3, 5, 6],
                      [1, 2, 4, 5]]),

            np.array([[5, 6, 0, 8, 9, 0, 0, 0, 0],
                      [4, 5, 6, 7, 8, 9, 0, 0, 0],
                      [0, 4, 5, 0, 7, 8, 0, 0, 0],

                      [2, 3, 0, 5, 6, 0, 8, 9, 0],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 0, 4, 5, 0, 7, 8],

                      [0, 0, 0, 2, 3, 0, 5, 6, 0],
                      [0, 0, 0, 1, 2, 3, 4, 5, 6],
                      [0, 0, 0, 0, 1, 2, 0, 4, 5]]),

            np.array(
                [[5, 6, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 4, 5, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0],

                 [2, 3, 0, 0, 5, 6, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0],
                 [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0],
                 [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0],
                 [0, 0, 1, 2, 0, 0, 4, 5, 0, 0, 7, 8, 0, 0, 0, 0],

                 [0, 0, 0, 0, 2, 3, 0, 0, 5, 6, 0, 0, 8, 9, 0, 0],
                 [0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0],
                 [0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9],
                 [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 4, 5, 0, 0, 7, 8],

                 [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 5, 6, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 4, 5]]),

            np.array([[5, 6, 0, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 4, 5, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                      [2, 3, 0, 0, 0, 5, 6, 0, 0, 0, 8, 9, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0, 0, 4, 5, 0, 0, 0, 7, 8,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                      [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 5, 6, 0, 0, 0,
                       8, 9, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0,
                       7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0,
                       0, 7, 8, 9, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6,
                       0, 0, 7, 8, 9, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 4, 5,
                       0, 0, 0, 7, 8, 0, 0, 0, 0, 0],

                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0,
                       5, 6, 0, 0, 0, 8, 9, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0,
                       4, 5, 6, 0, 0, 7, 8, 9, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
                       0, 4, 5, 6, 0, 0, 7, 8, 9, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
                       0, 0, 4, 5, 6, 0, 0, 7, 8, 9],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
                       0, 0, 0, 4, 5, 0, 0, 0, 7, 8],

                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       2, 3, 0, 0, 0, 5, 6, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 2, 3, 0, 0, 4, 5, 6, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 1, 2, 3, 0, 0, 4, 5, 6, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 2, 3, 0, 0, 4, 5, 6],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 1, 2, 0, 0, 0, 4, 5]])
        ]

        self._test_compute_weights(
            kernel_weights=np.array([[1, 2, 3],
                                     [4, 5, 6],
                                     [7, 8, 9]]),
            border_types=BorderType.PADDED,
            input_shapes=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            expected_weights=expected_weights
        )

    def test_connectivity_matrix_2d_odd_kernel_circular(self) -> None:
        """Tests whether computing weights works for 2D inputs with an odd
        sized kernel and CIRCULAR border type."""
        expected_weights = [
            np.array([[5]]),

            np.array([[5, 4, 2, 1],
                      [4, 5, 1, 2],

                      [2, 1, 5, 4],
                      [1, 2, 4, 5]]),

            np.array([[5, 6, 4, 8, 9, 7, 2, 3, 1],
                      [4, 5, 6, 7, 8, 9, 1, 2, 3],
                      [6, 4, 5, 9, 7, 8, 3, 1, 2],

                      [2, 3, 1, 5, 6, 4, 8, 9, 7],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [3, 1, 2, 6, 4, 5, 9, 7, 8],

                      [8, 9, 7, 2, 3, 1, 5, 6, 4],
                      [7, 8, 9, 1, 2, 3, 4, 5, 6],
                      [9, 7, 8, 3, 1, 2, 6, 4, 5]]),
            np.array(
                [[5, 6, 0, 4, 8, 9, 0, 7, 0, 0, 0, 0, 2, 3, 0, 1],
                 [4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0, 1, 2, 3, 0],
                 [0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0, 1, 2, 3],
                 [6, 0, 4, 5, 9, 0, 7, 8, 0, 0, 0, 0, 3, 0, 1, 2],

                 [2, 3, 0, 1, 5, 6, 0, 4, 8, 9, 0, 7, 0, 0, 0, 0],
                 [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0],
                 [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0],
                 [3, 0, 1, 2, 6, 0, 4, 5, 9, 0, 7, 8, 0, 0, 0, 0],

                 [0, 0, 0, 0, 2, 3, 0, 1, 5, 6, 0, 4, 8, 9, 0, 7],
                 [0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0],
                 [0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9],
                 [0, 0, 0, 0, 3, 0, 1, 2, 6, 0, 4, 5, 9, 0, 7, 8],

                 [8, 9, 0, 7, 0, 0, 0, 0, 2, 3, 0, 1, 5, 6, 0, 4],
                 [7, 8, 9, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0],
                 [0, 7, 8, 9, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6],
                 [9, 0, 7, 8, 0, 0, 0, 0, 3, 0, 1, 2, 6, 0, 4, 5]]),

            np.array([[5, 6, 0, 0, 4, 8, 9, 0, 0, 7, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 2, 3, 0, 0, 1],
                      [4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 1, 2, 3, 0, 0],
                      [0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 1, 2, 3, 0],
                      [0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
                      [6, 0, 0, 4, 5, 9, 0, 0, 7, 8, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 3, 0, 0, 1, 2],

                      [2, 3, 0, 0, 1, 5, 6, 0, 0, 4, 8, 9, 0, 0, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [3, 0, 0, 1, 2, 6, 0, 0, 4, 5, 9, 0, 0, 7, 8,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                      [0, 0, 0, 0, 0, 2, 3, 0, 0, 1, 5, 6, 0, 0, 4,
                       8, 9, 0, 0, 7, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0,
                       7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0,
                       0, 7, 8, 9, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6,
                       0, 0, 7, 8, 9, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 3, 0, 0, 1, 2, 6, 0, 0, 4, 5,
                       9, 0, 0, 7, 8, 0, 0, 0, 0, 0],

                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 1,
                       5, 6, 0, 0, 4, 8, 9, 0, 0, 7],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0,
                       4, 5, 6, 0, 0, 7, 8, 9, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
                       0, 4, 5, 6, 0, 0, 7, 8, 9, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
                       0, 0, 4, 5, 6, 0, 0, 7, 8, 9],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 2,
                       6, 0, 0, 4, 5, 9, 0, 0, 7, 8],

                      [8, 9, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       2, 3, 0, 0, 1, 5, 6, 0, 0, 4],
                      [7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 2, 3, 0, 0, 4, 5, 6, 0, 0],
                      [0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 1, 2, 3, 0, 0, 4, 5, 6, 0],
                      [0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 2, 3, 0, 0, 4, 5, 6],
                      [9, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       3, 0, 0, 1, 2, 6, 0, 0, 4, 5]])
        ]

        self._test_compute_weights(
            kernel_weights=np.array([[1, 2, 3],
                                     [4, 5, 6],
                                     [7, 8, 9]]),
            border_types=BorderType.CIRCULAR,
            input_shapes=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            expected_weights=expected_weights
        )

    def test_connectivity_matrix_2d_even_kernel_padded(self) -> None:
        """Tests whether computing weights works for 2D inputs with an even
        sized kernel and PADDED border type."""
        expected_weights = [
            np.array([[11]]),

            np.array([[11, 12, 15, 16],
                      [10, 11, 14, 15],

                      [7, 8, 11, 12],
                      [6, 7, 10, 11]]),

            np.array([[11, 12, 0, 15, 16, 0, 0, 0, 0],
                      [10, 11, 12, 14, 15, 16, 0, 0, 0],
                      [9, 10, 11, 13, 14, 15, 0, 0, 0],

                      [7, 8, 0, 11, 12, 0, 15, 16, 0],
                      [6, 7, 8, 10, 11, 12, 14, 15, 16],
                      [5, 6, 7, 9, 10, 11, 13, 14, 15],

                      [3, 4, 0, 7, 8, 0, 11, 12, 0],
                      [2, 3, 4, 6, 7, 8, 10, 11, 12],
                      [1, 2, 3, 5, 6, 7, 9, 10, 11]]),

            np.array([[11, 12, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0],
                      [10, 11, 12, 0, 14, 15, 16, 0, 0, 0, 0, 0, 0,
                       0, 0, 0],
                      [9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0,
                       0, 0, 0, 0],
                      [0, 9, 10, 11, 0, 13, 14, 15, 0, 0, 0, 0, 0,
                       0, 0, 0],

                      [7, 8, 0, 0, 11, 12, 0, 0, 15, 16, 0, 0, 0,
                       0, 0, 0],
                      [6, 7, 8, 0, 10, 11, 12, 0, 14, 15, 16, 0, 0,
                       0, 0, 0],
                      [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       0, 0, 0, 0],
                      [0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0,
                       0, 0, 0],

                      [3, 4, 0, 0, 7, 8, 0, 0, 11, 12, 0, 0, 15,
                       16, 0, 0],
                      [2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 0, 14,
                       15, 16, 0],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                       14, 15, 16],
                      [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13,
                       14, 15],

                      [0, 0, 0, 0, 3, 4, 0, 0, 7, 8, 0, 0, 11, 12,
                       0, 0],
                      [0, 0, 0, 0, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11,
                       12, 0],
                      [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12],
                      [0, 0, 0, 0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9,
                       10, 11]]),

            np.array([[11, 12, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [10, 11, 12, 0, 0, 14, 15, 16, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [9, 10, 11, 12, 0, 13, 14, 15, 16, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 9, 10, 11, 12, 0, 13, 14, 15, 16, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 9, 10, 11, 0, 0, 13, 14, 15, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                      [7, 8, 0, 0, 0, 11, 12, 0, 0, 0, 15, 16, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [6, 7, 8, 0, 0, 10, 11, 12, 0, 0, 14, 15, 16,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [5, 6, 7, 8, 0, 9, 10, 11, 12, 0, 13, 14, 15,
                       16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0, 13, 14,
                       15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 5, 6, 7, 0, 0, 9, 10, 11, 0, 0, 13,
                       14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                      [3, 4, 0, 0, 0, 7, 8, 0, 0, 0, 11, 12, 0, 0,
                       0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0],
                      [2, 3, 4, 0, 0, 6, 7, 8, 0, 0, 10, 11, 12, 0,
                       0, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0],
                      [1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12,
                       0, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0],
                      [0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11,
                       12, 0, 13, 14, 15, 16, 0, 0, 0, 0, 0],
                      [0, 0, 1, 2, 3, 0, 0, 5, 6, 7, 0, 0, 9, 10,
                       11, 0, 0, 13, 14, 15, 0, 0, 0, 0, 0],

                      [0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 7, 8, 0, 0, 0,
                       11, 12, 0, 0, 0, 15, 16, 0, 0, 0],
                      [0, 0, 0, 0, 0, 2, 3, 4, 0, 0, 6, 7, 8, 0, 0,
                       10, 11, 12, 0, 0, 14, 15, 16, 0, 0],
                      [0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0,
                       9, 10, 11, 12, 0, 13, 14, 15, 16, 0],
                      [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8,
                       0, 9, 10, 11, 12, 0, 13, 14, 15, 16],
                      [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 5, 6, 7,
                       0, 0, 9, 10, 11, 0, 0, 13, 14, 15],

                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0,
                       7, 8, 0, 0, 0, 11, 12, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 0, 0,
                       6, 7, 8, 0, 0, 10, 11, 12, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0,
                       5, 6, 7, 8, 0, 9, 10, 11, 12, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4,
                       0, 5, 6, 7, 8, 0, 9, 10, 11, 12],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
                       0, 0, 5, 6, 7, 0, 0, 9, 10, 11]])
        ]

        self._test_compute_weights(
            kernel_weights=np.array([[1, 2, 3, 4],
                                     [5, 6, 7, 8],
                                     [9, 10, 11, 12],
                                     [13, 14, 15, 16]]),
            border_types=BorderType.PADDED,
            input_shapes=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            expected_weights=expected_weights
        )

    def test_connectivity_matrix_2d_even_kernel_circular(self) -> None:
        """Tests whether computing weights works for 2D inputs with an even
        sized kernel and CIRCULAR border type."""
        expected_weights = [
            np.array([[11]]),

            np.array([[11, 10, 7, 6],
                      [10, 11, 6, 7],
                      [7, 6, 11, 10],
                      [6, 7, 10, 11]]),

            np.array([[11, 12, 10, 15, 16, 14, 7, 8, 6],
                      [10, 11, 12, 14, 15, 16, 6, 7, 8],
                      [12, 10, 11, 16, 14, 15, 8, 6, 7],

                      [7, 8, 6, 11, 12, 10, 15, 16, 14],
                      [6, 7, 8, 10, 11, 12, 14, 15, 16],
                      [8, 6, 7, 12, 10, 11, 16, 14, 15],

                      [15, 16, 14, 7, 8, 6, 11, 12, 10],
                      [14, 15, 16, 6, 7, 8, 10, 11, 12],
                      [16, 14, 15, 8, 6, 7, 12, 10, 11]]),

            np.array([[11, 12, 9, 10, 15, 16, 13, 14, 3, 4, 1, 2,
                       7, 8, 5, 6],
                      [10, 11, 12, 9, 14, 15, 16, 13, 2, 3, 4, 1,
                       6, 7, 8, 5],
                      [9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4,
                       5, 6, 7, 8],
                      [12, 9, 10, 11, 16, 13, 14, 15, 4, 1, 2, 3,
                       8, 5, 6, 7],

                      [7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14,
                       3, 4, 1, 2],
                      [6, 7, 8, 5, 10, 11, 12, 9, 14, 15, 16, 13,
                       2, 3, 4, 1],
                      [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       1, 2, 3, 4],
                      [8, 5, 6, 7, 12, 9, 10, 11, 16, 13, 14, 15,
                       4, 1, 2, 3],

                      [3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15,
                       16, 13, 14],
                      [2, 3, 4, 1, 6, 7, 8, 5, 10, 11, 12, 9, 14,
                       15, 16, 13],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                       14, 15, 16],
                      [4, 1, 2, 3, 8, 5, 6, 7, 12, 9, 10, 11, 16,
                       13, 14, 15],

                      [15, 16, 13, 14, 3, 4, 1, 2, 7, 8, 5, 6, 11,
                       12, 9, 10],
                      [14, 15, 16, 13, 2, 3, 4, 1, 6, 7, 8, 5, 10,
                       11, 12, 9],
                      [13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                       10, 11, 12],
                      [16, 13, 14, 15, 4, 1, 2, 3, 8, 5, 6, 7, 12,
                       9, 10, 11]]),

            np.array([[11, 12, 0, 9, 10, 15, 16, 0, 13, 14, 0, 0,
                       0, 0, 0, 3, 4, 0, 1, 2, 7, 8, 0, 5, 6],
                      [10, 11, 12, 0, 9, 14, 15, 16, 0, 13, 0, 0,
                       0, 0, 0, 2, 3, 4, 0, 1, 6, 7, 8, 0, 5],
                      [9, 10, 11, 12, 0, 13, 14, 15, 16, 0, 0, 0,
                       0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0],
                      [0, 9, 10, 11, 12, 0, 13, 14, 15, 16, 0, 0,
                       0, 0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8],
                      [12, 0, 9, 10, 11, 16, 0, 13, 14, 15, 0, 0,
                       0, 0, 0, 4, 0, 1, 2, 3, 8, 0, 5, 6, 7],

                      [7, 8, 0, 5, 6, 11, 12, 0, 9, 10, 15, 16, 0,
                       13, 14, 0, 0, 0, 0, 0, 3, 4, 0, 1, 2],
                      [6, 7, 8, 0, 5, 10, 11, 12, 0, 9, 14, 15, 16,
                       0, 13, 0, 0, 0, 0, 0, 2, 3, 4, 0, 1],
                      [5, 6, 7, 8, 0, 9, 10, 11, 12, 0, 13, 14, 15,
                       16, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0],
                      [0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0, 13, 14,
                       15, 16, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
                      [8, 0, 5, 6, 7, 12, 0, 9, 10, 11, 16, 0, 13,
                       14, 15, 0, 0, 0, 0, 0, 4, 0, 1, 2, 3],

                      [3, 4, 0, 1, 2, 7, 8, 0, 5, 6, 11, 12, 0, 9,
                       10, 15, 16, 0, 13, 14, 0, 0, 0, 0, 0],
                      [2, 3, 4, 0, 1, 6, 7, 8, 0, 5, 10, 11, 12, 0,
                       9, 14, 15, 16, 0, 13, 0, 0, 0, 0, 0],
                      [1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12,
                       0, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0],
                      [0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11,
                       12, 0, 13, 14, 15, 16, 0, 0, 0, 0, 0],
                      [4, 0, 1, 2, 3, 8, 0, 5, 6, 7, 12, 0, 9, 10,
                       11, 16, 0, 13, 14, 15, 0, 0, 0, 0, 0],

                      [0, 0, 0, 0, 0, 3, 4, 0, 1, 2, 7, 8, 0, 5, 6,
                       11, 12, 0, 9, 10, 15, 16, 0, 13, 14],
                      [0, 0, 0, 0, 0, 2, 3, 4, 0, 1, 6, 7, 8, 0, 5,
                       10, 11, 12, 0, 9, 14, 15, 16, 0, 13],
                      [0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0,
                       9, 10, 11, 12, 0, 13, 14, 15, 16, 0],
                      [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8,
                       0, 9, 10, 11, 12, 0, 13, 14, 15, 16],
                      [0, 0, 0, 0, 0, 4, 0, 1, 2, 3, 8, 0, 5, 6, 7,
                       12, 0, 9, 10, 11, 16, 0, 13, 14, 15],

                      [15, 16, 0, 13, 14, 0, 0, 0, 0, 0, 3, 4, 0,
                       1, 2, 7, 8, 0, 5, 6, 11, 12, 0, 9, 10],
                      [14, 15, 16, 0, 13, 0, 0, 0, 0, 0, 2, 3, 4,
                       0, 1, 6, 7, 8, 0, 5, 10, 11, 12, 0, 9],
                      [13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 1, 2, 3,
                       4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0],
                      [0, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 1, 2,
                       3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12],
                      [16, 0, 13, 14, 15, 0, 0, 0, 0, 0, 4, 0, 1,
                       2, 3, 8, 0, 5, 6, 7, 12, 0, 9, 10, 11]])
        ]

        self._test_compute_weights(
            kernel_weights=np.array([[1, 2, 3, 4],
                                     [5, 6, 7, 8],
                                     [9, 10, 11, 12],
                                     [13, 14, 15, 16]]),
            border_types=BorderType.CIRCULAR,
            input_shapes=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            expected_weights=expected_weights
        )


class TestReduceAlongDiagonal(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a ReduceAlongDiagonal operation can be instantiated."""
        diag_op = ReduceAlongDiagonal()
        self.assertIsInstance(diag_op, ReduceAlongDiagonal)

    def test_compute_weights_directly(self) -> None:
        """Tests whether computing weights produces the expected result for
        different shapes. This directly compares against small predetermined
        weight matrices."""
        shapes = [(3,), (4,), (2, 2)]
        expected_weights = [
            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        ]

        for shape, expected in zip(shapes, expected_weights):
            diag_op = ReduceAlongDiagonal()
            diag_op.configure(shape + shape)

            computed = diag_op.compute_weights()

            self.assertTrue(np.array_equal(computed, expected))

    def test_compute_weights_indirectly(self) -> None:
        """Tests computing weights indirectly by multiplying the computed
        weight matrices with simple matrices. This is to be able to test
        higher dimensional cases where direct testing of the weight matrices
        would require pasting very large matrices into the source code."""
        # go through a 1D, 2D, and 3D case
        shapes = [(3,), (3, 4), (3, 4, 5)]

        for shape in shapes:
            # Construct a mock spike-input from a 'transformation DNF'
            # that is twice the dimensionality of the input shape
            spike_input_shape = shape + shape
            spike_input = np.zeros(spike_input_shape,
                                   dtype=np.int32)
            # Generate an index such that the second element of the last
            # dimension is 1:
            # e.g., a 1D spike input would be: [0, 1, 0, 0],
            # a 2D spike input would be: [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
            idx = np.zeros(spike_input.ndim, dtype=np.int32)
            idx[-1] = 1
            spike_input[tuple(idx)] = 1
            # Flatten the input to enable matrix multiplication with the
            # connectivity weight matrix
            spike_input = spike_input.ravel()

            # Create and configure the operation
            diag_op = ReduceAlongDiagonal()
            diag_op.configure(spike_input_shape)

            # Use the operation to compute the connectivity weight matrix
            weights = diag_op.compute_weights()

            # Compute the output of the synaptic connections if the
            # connectivity matrix were applied to the spike input
            computed = np.matmul(weights, spike_input)

            # Generate the expected output of the synaptic connections;
            # the shape along every dimension must be twice the input minus 1,
            # where all elements are zero except for the second entry of the
            # last dimension
            expected_shape = tuple(np.array(shape) * 2 - 1)
            expected = np.zeros(expected_shape, dtype=np.int32)
            expected_idx = np.zeros(expected.ndim, dtype=np.int32)
            expected_idx[-1] = 1
            expected[tuple(expected_idx)] = 1
            expected = expected.ravel()

            self.assertTrue(np.array_equal(computed, expected))


class TestExpandAlongDiagonal(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a ExpandAlongDiagonal operation can be instantiated."""
        diag_op = ExpandAlongDiagonal()
        self.assertIsInstance(diag_op, ExpandAlongDiagonal)

    def test_compute_weights_directly(self) -> None:
        """Tests whether computing weights produces the expected result for
        different shapes. This directly compares against small predetermined
        weight matrices."""
        shapes = [(3,), (5,), (3, 3)]
        expected_weights = [
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        ]

        for shape, expected in zip(shapes, expected_weights):
            diag_op = ExpandAlongDiagonal()
            diag_op.configure(shape)

            computed = diag_op.compute_weights()

            self.assertTrue(np.array_equal(computed, expected))

    def test_compute_weights_indirectly(self) -> None:
        """Tests computing weights indirectly by multiplying the computed
        weight matrices with simple matrices. This is to be able to test
        higher dimensional cases where direct testing of the weight matrices
        would require pasting very large matrices into the source code."""
        # Go through a 1D, 2D, and 3D case
        shapes = [(5,), (5, 7), (5, 7, 9)]

        for shape in shapes:
            # Construct a mock spike-input from a 'relational DNF'
            spike_input = np.zeros(shape, dtype=np.int32)
            # Generate an index such that the second element of the last
            # dimension is 1:
            # e.g., a 1D spike input would be: [0, 1, 0, 0],
            # a 2D spike input would be: [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
            idx = np.zeros(spike_input.ndim, dtype=np.int32)
            idx[-1] = 1
            spike_input[tuple(idx)] = 1

            # Create and configure the operation
            diag_op = ExpandAlongDiagonal()
            diag_op.configure(shape)

            # Use the operation to compute the connectivity weight matrix
            weights = diag_op.compute_weights()

            # Compute the output of the synaptic connections if the
            # connectivity matrix were applied to the spike input
            computed = np.matmul(weights, spike_input.ravel())

            # Generate the expected output of the synaptic connections;
            # the shape along every dimension must be (input+1)*2,
            # where all elements are zero except for the "line" projected
            # diagonally through the matrix
            half_shape = (np.array(shape) + 1) / 2
            half_shape = tuple(half_shape.astype(int))
            expected_shape = half_shape + half_shape
            expected = np.zeros(expected_shape, dtype=np.int32)
            # Get the position of the spike input
            x = np.argwhere(spike_input)[0]
            for p in np.ndindex(*half_shape):
                p = np.array(p)

                # Set expected[x-p, p] = 1, where 0 <= x-p < half_shape
                d = x - p
                if np.all(0 <= d) and np.all(d < np.array(half_shape)):
                    i = tuple(np.concatenate([d, p]))
                    expected[i] = 1
            expected = expected.ravel()

            self.assertTrue(np.array_equal(computed, expected))


class TestFlip(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a Flip operation can be instantiated."""
        flip_op = Flip()
        self.assertIsInstance(flip_op, Flip)

    def test_operation_defaults_to_flipping_all_dims(self) -> None:
        """Tests whether the operation flips all dimensions when no
        'dims' attribute is supplied."""
        shape = (3, 4, 5)
        spike_input = np.zeros(shape, dtype=np.int32)
        spike_input[1, 1, 1] = 1

        flip_op = Flip()
        flip_op.configure(input_shape=shape)
        weights = flip_op.compute_weights()
        computed = np.matmul(spike_input.ravel(), weights)
        computed = computed.reshape(shape)

        expected = np.flip(spike_input, axis=(0, 1, 2))

        self.assertTrue(np.array_equal(computed, expected))

    def test_flipping_specific_dimensions(self) -> None:
        """Tests whether the computed weight matrix flips the input along the
        specified dimensions."""

        def all_sub_sets(_set: ty.List) -> ty.Iterator:
            """Creates a list of all possible subsets of a given set."""
            sub_sets = it.chain.from_iterable(
                it.combinations(_set, length) for length in range(len(_set) + 1)
            )
            return sub_sets

        for shape in [(5,), (4, 5), (3, 4, 5)]:
            # Construct a mock spike-input
            spike_input = np.zeros(shape, dtype=np.int32)
            # Generate an index such that the second element of the last
            # dimension is 1:
            # e.g., a 1D spike input would be: [0, 1, 0, 0],
            # a 2D spike input would be: [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
            idx = np.zeros(spike_input.ndim, dtype=np.int32)
            idx[-1] = 1
            spike_input[tuple(idx)] = 1

            # Go through all subsets of all indices of the shape
            dim_indices = range(len(shape))
            for dims in all_sub_sets(dim_indices):
                # Set up a Flip operation
                flip_op = Flip(dims=dims)
                flip_op.configure(input_shape=shape)
                # Compute it's connectivity matrix
                weights = flip_op.compute_weights()
                # Apply the connectivity matrix to the mock spike-input
                computed = np.matmul(spike_input.ravel(), weights)
                computed = computed.reshape(shape)

                # Compute what we expect the Flip operation to do
                expected = np.flip(spike_input, axis=dims)

                # Compare the computed with the expected output
                self.assertTrue(np.array_equal(computed, expected))


if __name__ == '__main__':
    unittest.main()

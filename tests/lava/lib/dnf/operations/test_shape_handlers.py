# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty

from lava.lib.dnf.operations.shape_handlers import (
    AbstractShapeHandler,
    KeepShapeHandler,
    ReduceDimsHandler,
    ReshapeHandler,
    ExpandDimsHandler,
    ReorderHandler)
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError


class MockShapeHandler(AbstractShapeHandler):
    """Mock shape handler for testing"""
    def __init__(self) -> None:
        super().__init__()
        self.args_validated = False
        self.input_shape_validated = False

    def _compute_output_shape(self) -> None:
        self._output_shape = self._input_shape

    def _validate_args(self) -> None:
        self.args_validated = True

    def _validate_input_shape(self, input_shape: ty.Tuple[int, ...]) -> None:
        self.input_shape_validated = True


class TestAbstractShapeHandler(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a MockShapeHandler can be instantiated."""
        sh = MockShapeHandler()
        self.assertIsInstance(sh, MockShapeHandler)

    def test_assert_configured_raises_error_when_not_configured(self) -> None:
        """Tests whether assert_configured() raises an assertion error when
        the ShapeHandler is not yet configured."""
        sh = MockShapeHandler()
        self.assertRaises(AssertionError, sh.assert_configured)

    def test_configure_works(self) -> None:
        """Tests whether configure() validates and sets the input shape,
        validates any args, and computes the output shape."""
        sh = MockShapeHandler()
        input_shape = (2, 4)
        sh.configure(input_shape)

        self.assertTrue(sh.input_shape_validated)
        self.assertEqual(sh._input_shape, input_shape)
        self.assertTrue(sh.args_validated)
        self.assertEqual(sh._output_shape, input_shape)

    def test_output_shape_getter(self) -> None:
        """Tests the getter for output shape"""
        sh = MockShapeHandler()
        output_shape = (2, 4)
        sh._output_shape = output_shape
        self.assertEqual(sh.output_shape, output_shape)

    def test_input_shape_getter(self) -> None:
        """Tests the getter for input shape"""
        sh = MockShapeHandler()
        input_shape = (2, 4)
        sh._input_shape = input_shape
        self.assertEqual(sh.input_shape, input_shape)


class TestKeepShapeHandler(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a KeepShapeHandler can be instantiated."""
        sh = KeepShapeHandler()
        self.assertIsInstance(sh, KeepShapeHandler)

    def test_compute_output_shape(self) -> None:
        """Tests whether the output shape is set correctly."""
        sh = KeepShapeHandler()
        input_shape = (2, 4)
        sh.configure(input_shape=input_shape)
        self.assertEqual(sh.output_shape, input_shape)


class TestExpandDimsHandler(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a ExpandDimsHandler can be instantiated."""
        sh = ExpandDimsHandler(new_dims_shape=6)
        self.assertIsInstance(sh, ExpandDimsHandler)

    def test_compute_output_shape_expand_one_dim_with_int_argument(self)\
            -> None:
        """Tests whether the output shape is set correctly when a single
        dimension is added, using an integer to specify the shape."""
        sh = ExpandDimsHandler(new_dims_shape=6)
        sh.configure(input_shape=(2, 4))
        self.assertEqual(sh.output_shape, (2, 4, 6))

    def test_compute_output_shape_expand_multiple_dims(self) -> None:
        """Tests whether the output shape is set correctly when multiple
         dimensions are added."""
        sh = ExpandDimsHandler(new_dims_shape=(6, 8))
        sh.configure(input_shape=(2,))
        self.assertEqual(sh.output_shape, (2, 6, 8))

    def test_compute_output_shape_expand_from_0d(self) -> None:
        """Tests whether the output shape is set correctly when expanding
        from 0D to 1D."""
        sh = ExpandDimsHandler(new_dims_shape=(10,))
        sh.configure(input_shape=(1,))
        self.assertEqual(sh.output_shape, (10,))

    def test_negative_shape_values_raise_error(self) -> None:
        """Tests whether an error is raised when <new_dims_shape> contains a
        negative value."""
        sh = ExpandDimsHandler(new_dims_shape=(-6,))
        with self.assertRaises(ValueError):
            sh.configure(input_shape=(2, 4))

    def test_zero_shape_values_raise_error(self) -> None:
        """Tests whether an error is raised when <new_dims_shape> contains a
        zero."""
        sh = ExpandDimsHandler(new_dims_shape=(0,))
        with self.assertRaises(ValueError):
            sh.configure(input_shape=(2, 4))

    def test_empty_new_dims_shape_raises_error(self) -> None:
        """Tests whether an error is raised when <new_dims_shape> is empty."""
        sh = ExpandDimsHandler(new_dims_shape=())
        with self.assertRaises(ValueError):
            sh.configure(input_shape=(2, 4))

    def test_output_shape_larger_than_dim_3_raises_error(self) -> None:
        """Tests whether an error is raised when the computed output shape is
        larger than 3."""
        sh = ExpandDimsHandler(new_dims_shape=(6, 8))
        with self.assertRaises(NotImplementedError):
            sh.configure(input_shape=(2, 4))


class TestReduceDimsHandler(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a ReduceDimsHandler can be instantiated."""
        sh = ReduceDimsHandler(reduce_dims=1)
        self.assertIsInstance(sh, ReduceDimsHandler)

    def test_compute_output_shape_remove_one_dim(self) -> None:
        """Tests whether the output shape is set correctly when a single
        dimension is removed."""
        sh = ReduceDimsHandler(reduce_dims=1)
        sh.configure(input_shape=(2, 4))
        self.assertEqual(sh.output_shape, (2,))

    def test_compute_output_shape_negative_index(self) -> None:
        """Tests whether the output shape is set correctly when a single
        dimension is removed, using a negative index."""
        sh = ReduceDimsHandler(reduce_dims=-1)
        sh.configure(input_shape=(2, 4))
        self.assertEqual(sh.output_shape, (2,))

    def test_compute_output_shape_remove_multiple_dims(self) -> None:
        """Tests whether the output shape is set correctly when multiple
        dimensions are removed."""
        sh = ReduceDimsHandler(reduce_dims=(0, -1))
        sh.configure(input_shape=(2, 3, 4, 5))
        self.assertEqual(sh.output_shape, (3, 4))

    def test_compute_output_shape_remove_all(self) -> None:
        """Tests whether the output shape is set correctly when all
        dimensions are removed."""
        sh = ReduceDimsHandler(reduce_dims=(0, 1, 2))
        sh.configure(input_shape=(2, 3, 4))
        self.assertEqual(sh.output_shape, (1,))

    def test_order_of_reduce_dims_does_not_impact_result(self) -> None:
        """Tests whether the order of <reduce_dims> does not matter."""
        input_shape = (3, 4, 5)
        sh1 = ReduceDimsHandler(reduce_dims=(1, 2))
        sh1.configure(input_shape=input_shape)

        sh2 = ReduceDimsHandler(reduce_dims=(1, 2))
        sh2.configure(input_shape=input_shape)

        self.assertTrue(np.array_equal(sh1.output_shape,
                                       sh2.output_shape))

    def test_reduce_dims_with_out_of_bounds_index_raises_error(self) -> None:
        """Tests whether an error is raised when <reduce_dims> contains an
        index that is out of bounds for the input shape."""
        with self.assertRaises(IndexError):
            sh = ReduceDimsHandler(reduce_dims=2)
            sh.configure(input_shape=(2, 4))

    def test_reduce_dims_with_negative_out_of_bounds_index_raises_error(self)\
            -> None:
        """Tests whether an error is raised when <reduce_dims> contains a
        negative index that is out of bounds for the input shape."""
        with self.assertRaises(IndexError):
            sh = ReduceDimsHandler(reduce_dims=-3)
            sh.configure(input_shape=(2, 4))

    def test_empty_reduce_dims_raises_error(self) -> None:
        """Tests whether an error is raised when <reduce_dims> is an empty
        tuple."""
        with self.assertRaises(ValueError):
            sh = ReduceDimsHandler(reduce_dims=())
            sh.configure(input_shape=(2, 4))

    def test_reduce_dims_with_too_many_entries_raises_error(self) -> None:
        """Tests whether an error is raised when <reduce_dims> has more
        elements than the dimensionality of the input."""
        with self.assertRaises(ValueError):
            sh = ReduceDimsHandler(reduce_dims=(0, 0))
            sh.configure(input_shape=(4,))

    def test_zero_dimensional_input_shape_raises_error(self) -> None:
        """Tests whether an error is raised when the input shape is already
        zero-dimensional."""
        with self.assertRaises(MisconfiguredOpError):
            sh = ReduceDimsHandler(reduce_dims=0)
            sh.configure(input_shape=(1,))


class TestReshapeHandler(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a ReshapeHandler can be instantiated."""
        sh = ReshapeHandler(output_shape=(8,))
        self.assertIsInstance(sh, ReshapeHandler)

    def test_compute_output_shape(self) -> None:
        """Tests whether the output shape is set correctly."""
        output_shape = (8,)
        sh = ReshapeHandler(output_shape=output_shape)
        sh.configure(input_shape=(2, 4))
        self.assertEqual(sh.output_shape, output_shape)

    def test_compute_output_shape_with_incorrect_num_neurons_raises_error(self)\
            -> None:
        """Tests whether an error is raised when the number of neurons in
        the input and output shape does not match."""
        with self.assertRaises(MisconfiguredOpError):
            output_shape = (9,)
            sh = ReshapeHandler(output_shape=output_shape)
            sh.configure(input_shape=(2, 4))


class TestReorderHandlerShapeHandler(unittest.TestCase):
    def test_init(self) -> None:
        """Tests whether a ReorderHandler can be instantiated."""
        sh = ReorderHandler(order=(1, 0))
        self.assertIsInstance(sh, ReorderHandler)

    def test_order_with_more_elements_than_input_raises_error(self) -> None:
        """Tests whether an error is raised when the specified new <order>
        has more elements than the number of input dimensions."""
        sh = ReorderHandler(order=(1, 0, 2))
        with self.assertRaises(MisconfiguredOpError):
            sh.configure(input_shape=(2, 2))

    def test_order_with_less_elements_than_input_raises_error(self) -> None:
        """Tests whether an error is raised when the specified new <order>
        has less elements than the number of input dimensions."""
        sh = ReorderHandler(order=(1,))
        with self.assertRaises(MisconfiguredOpError):
            sh.configure(input_shape=(2, 2))

    def test_negative_order_index_within_bounds_works(self) -> None:
        """Tests whether indices in <order> can be specified with negative
        numbers."""
        sh = ReorderHandler(order=(0, -2))
        sh.configure(input_shape=(2, 2))

    def test_order_index_out_of_bounds_raises_error(self) -> None:
        """Tests whether an error is raised when an index in <order> is
        larger than the dimensionality of the input."""
        sh = ReorderHandler(order=(0, 2))
        with self.assertRaises(IndexError):
            sh.configure(input_shape=(2, 2))

    def test_negative_order_index_out_of_bounds_raises_error(self) -> None:
        """Tests whether an error is raised when an index in <order> is
        negative and out of bounds."""
        sh = ReorderHandler(order=(0, -3))
        with self.assertRaises(IndexError):
            sh.configure(input_shape=(2, 2))

    def test_input_dimensionality_0_raises_error(self) -> None:
        """Tests whether an error is raised when the input dimensionality is
        0, in which case reordering does not make sense."""
        sh = ReorderHandler(order=(0,))
        with self.assertRaises(MisconfiguredOpError):
            sh.configure(input_shape=(1,))

    def test_input_dimensionality_1_raises_error(self) -> None:
        """Tests whether an error is raised when the input dimensionality is
        1, in which case reordering does not make sense."""
        sh = ReorderHandler(order=(0,))
        with self.assertRaises(MisconfiguredOpError):
            sh.configure(input_shape=(5,))

    def test_reordering_2d(self) -> None:
        """Tests whether reordering a two-dimensional input works."""
        sh = ReorderHandler(order=(1, 0))
        sh.configure(input_shape=(0, 1))
        self.assertEqual(sh.output_shape, (1, 0))

    def test_reordering_3d(self) -> None:
        """Tests whether reordering a three-dimensional input works."""
        sh = ReorderHandler(order=(1, 2, 0))
        sh.configure(input_shape=(0, 1, 2))
        self.assertEqual(sh.output_shape, (1, 2, 0))

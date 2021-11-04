# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import itertools

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess

from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.connect.exceptions import MissingOpError, DuplicateOpError
from lava.lib.dnf.operations.operations import AbstractOperation
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError
from lava.lib.dnf.utils.convenience import num_neurons, num_dims


class MockProcess(AbstractProcess):
    """Mock Process with an InPort and OutPort"""
    def __init__(self, shape=(1,)):
        super().__init__()
        self.a_in = InPort(shape)
        self.s_out = OutPort(shape)


class MockOperation(AbstractOperation):
    """Mock Operation that generates an identity matrix as weights"""
    def __init__(self, input_shape=(1,), output_shape=(1,)):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _compute_weights(self):
        return np.eye(num_neurons(self.output_shape),
                      num_neurons(self.input_shape),
                      dtype=np.int32)

    def _validate_configuration(self):
        if self.input_shape != self.output_shape:
            raise MisconfiguredOpError("input_shape must be equal to "
                                       "output_shape")


class MockProjection(MockOperation):
    """Mock Projection operation"""
    def __init__(self):
        super().__init__()
        self.changes_dim = True

    def _validate_configuration(self) -> bool:
        if num_dims(self.input_shape) == num_dims(self.output_shape):
            raise MisconfiguredOpError("input dimensionality must not be "
                                       "equal to output dimensionality")


class MockResize(MockOperation):
    """Mock Projection operation"""
    def __init__(self):
        super().__init__()
        self.changes_size = True

    def _validate_configuration(self) -> bool:
        valid = False

        if num_dims(self.input_shape) == num_dims(self.output_shape):
            for si, so in zip(self.input_shape, self.output_shape):
                if si != so:
                    valid = True

        return valid


class MockReorder(MockOperation):
    """Mock Reorder operation"""
    def __init__(self):
        super().__init__()
        self.reorders_shape = True

    def _validate_configuration(self) -> bool:
        valid = False

        if num_dims(self.input_shape) == num_dims(self.output_shape):
            if self.output_shape in \
                    list(itertools.permutations(self.input_shape)):
                valid = True

        return valid


class TestConnect(unittest.TestCase):
    def test_connect_function_exists_and_is_callable(self):
        """Tests whether the connect function exists and is callable."""
        import lava
        self.assertTrue(callable(getattr(lava.lib.dnf.connect.connect,
                                         'connect')))

    def test_connecting_source_and_destination(self):
        """Tests connecting a source Process to a destination Process."""
        # create mock processes and an operation to connect
        source = MockProcess(shape=(1, 2, 3))
        destination = MockProcess(shape=(1, 2, 3))
        op = MockOperation()

        # connect source to target
        connections = connect(source.s_out, destination.a_in, ops=[op])

        # check whether the connect function returns a process
        self.assertIsInstance(connections, AbstractProcess)

        # check whether 'source' is connected to 'connections'
        src_op = source.out_ports.s_out
        con_ip = connections.in_ports.s_in
        self.assertEqual(src_op.get_dst_ports(), [con_ip])
        self.assertEqual(con_ip.get_src_ports(), [src_op])

        # check whether 'connections' is connected to 'target'
        con_op = connections.out_ports.a_out
        dst_op = destination.in_ports.a_in
        self.assertEqual(con_op.get_dst_ports(), [dst_op])
        self.assertEqual(dst_op.get_src_ports(), [con_op])

    def test_empty_operations_list_raises_value_error(self):
        """Tests whether an empty <ops> argument raises a value error."""
        with self.assertRaises(ValueError):
            connect(MockProcess().s_out, MockProcess().a_in, ops=[])

    def test_ops_list_containing_invalid_type_raises_type_error(self):
        """Tests whether the type of all elements in <ops> is validated."""
        class NotAnOperation:
            pass

        with self.assertRaises(TypeError):
            connect(MockProcess().s_out,
                    MockProcess().a_in,
                    ops=[MockOperation(), NotAnOperation()])

    def test_single_operation_can_be_passed_in_as_ops(self):
        """Tests whether a single operation is wrapped into a list."""
        connect(MockProcess().s_out,
                MockProcess().a_in,
                ops=MockOperation())

    def test_duplicate_operations_changing_dim_raises_error(self):
        """Tests whether an exception is raised when the user specifies
        multiple operations that change the dimensionality."""
        with self.assertRaises(DuplicateOpError) as context:
            connect(MockProcess((2, 4)).s_out,
                    MockProcess((2,)).a_in,
                    ops=[MockProjection(), MockProjection()])
        self.assertEqual(context.exception.duplicate_op, "changes_dim")

    def test_duplicate_operations_changing_size_raises_error(self):
        """Tests whether an exception is raised when the user specifies
        multiple operations that change the size."""
        with self.assertRaises(DuplicateOpError) as context:
            connect(MockProcess((5,)).s_out,
                    MockProcess((6,)).a_in,
                    ops=[MockResize(), MockResize()])
        self.assertEqual(context.exception.duplicate_op, "changes_size")

    def test_duplicate_operations_reordering_shape_raises_error(self):
        """Tests whether an exception is raised when the user specifies
        multiple operations that reorder the shape."""
        with self.assertRaises(DuplicateOpError) as context:
            connect(MockProcess((5, 3)).s_out,
                    MockProcess((3, 5)).a_in,
                    ops=[MockReorder(), MockReorder()])
        self.assertEqual(context.exception.duplicate_op, "reorders_shape")

    def test_shape_mismatch_with_correct_operation(self):
        """Tests whether validation passes when the shape of the source
        OutPort does not match the destination InPort and a Projection
        operation is specified."""
        connect(MockProcess((2, 4)).s_out,
                MockProcess((2,)).a_in,
                ops=[MockProjection()])

    def test_shape_mismatch_without_correct_operation_raises_error(self):
        """Tests whether an error is raised when the shape of the source
        OutPort does not match the destination InPort and a Projection is
        required but missing."""
        with self.assertRaises(MissingOpError) as context:
            connect(MockProcess((2, 4)).s_out,
                    MockProcess((2,)).a_in,
                    ops=[MockOperation()])
        self.assertEqual(context.exception.missing_op, "changes_dim")

    def test_size_mismatch_with_correct_operation(self):
        """Tests whether validation passes when the sizes of the source
        OutPort do not match the destination InPort and a Resize
        operation is specified."""
        connect(MockProcess((5,)).s_out,
                MockProcess((6,)).a_in,
                ops=[MockResize()])

    def test_size_mismatch_without_correct_operation_raises_error(self):
        """Tests whether an error is raised when the shape of the source
        OutPort does not match the destination InPort and a Resize is
        required but missing."""
        with self.assertRaises(MissingOpError) as context:
            connect(MockProcess((5,)).s_out,
                    MockProcess((6,)).a_in,
                    ops=[MockOperation()])
        self.assertEqual(context.exception.missing_op, "changes_size")

    def test_shape_order_mismatch_with_correct_operation(self):
        """Tests whether validation passes when the shape-order of the source
        OutPort does not match the destination InPort and a Reorder
        operation is specified."""
        connect(MockProcess((5, 3)).s_out,
                MockProcess((3, 5)).a_in,
                ops=[MockReorder()])

    def test_shape_order_mismatch_without_correct_operation_raises_error(self):
        """Tests whether an error is raised when the shape of the source
        OutPort does not match the destination InPort and a Reorder is
        required but missing."""
        with self.assertRaises(MissingOpError) as context:
            connect(MockProcess((5, 3)).s_out,
                    MockProcess((3, 5)).a_in,
                    ops=[MockOperation()])
        self.assertEqual(context.exception.missing_op, "reorders_shape")

    def test_invalid_op_configuration_raises_value_error(self):
        """Tests whether an error is raised when an operation has an invalid
        configuration."""
        class InvalidOperation(MockOperation):
            """Operation whose configuration is always invalid"""
            def _validate_configuration(self):
                raise MisconfiguredOpError("misconfigured")

        with self.assertRaises(MisconfiguredOpError):
            connect(MockProcess().s_out,
                    MockProcess().a_in,
                    ops=[InvalidOperation()])

    def test_combining_multiple_ops_that_do_not_change_shape(self):
        """Tests whether multiple operations can be specified that do not
        change the shape."""
        connect(MockProcess((5, 3)).s_out,
                MockProcess((5, 3)).a_in,
                ops=[MockOperation(), MockOperation(), MockOperation()])

    def test_multiple_non_changing_ops_and_one_that_changes_dim(self):
        """Tests whether an operation that changes the dimensionality
        can be combined with multiple operations that do not change the
        shape."""
        connect(MockProcess((5, 3)).s_out,
                MockProcess((5,)).a_in,
                ops=[MockOperation(), MockProjection(), MockOperation()])

    def test_multiple_non_changing_ops_and_one_that_changes_size(self):
        """Tests whether an operation that changes the size
        can be combined with multiple operations that do not change the
        shape."""
        connect(MockProcess((5, 3)).s_out,
                MockProcess((5, 6)).a_in,
                ops=[MockOperation(), MockResize(), MockOperation()])

    def test_multiple_non_changing_ops_and_one_that_reorders_shape(self):
        """Tests whether an operation that reorders shape
        can be combined with multiple operations that do not change the
        shape."""
        connect(MockProcess((5, 3)).s_out,
                MockProcess((3, 5)).a_in,
                ops=[MockOperation(), MockReorder(), MockOperation()])

    def test_multiple_ops_that_make_changes_raises_not_impl_error(self):
        """Tests whether a NotImplementedError is raised when multiple
         operations are specified that each make a change to the shape."""
        with self.assertRaises(NotImplementedError):
            connect(MockProcess((5, 3)).s_out,
                    MockProcess((2, 5)).a_in,
                    ops=[MockProjection(), MockResize()])

    def test_weights_from_multiple_ops_get_multiplied(self):
        """Tests whether compute_weights() multiplies the weights that are
        produced by all specified operations."""

        class MockOpWeights(MockOperation):
            """Mock Operation that generates an identity matrix with a given
            weight."""
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def _compute_weights(self):
                return np.eye(num_neurons(self.input_shape),
                              num_neurons(self.output_shape),
                              dtype=np.int32) * self.weight

        shape = (5, 3)
        w1 = 2
        w2 = 4

        conn = connect(MockProcess(shape).s_out,
                       MockProcess(shape).a_in,
                       ops=[MockOpWeights(weight=w1),
                            MockOpWeights(weight=w2)])

        computed_weights = conn.weights.get()
        expected_weights = np.eye(num_neurons(shape),
                                  num_neurons(shape),
                                  dtype=np.int32) * w1 * w2

        self.assertTrue(np.array_equal(computed_weights, expected_weights))


if __name__ == '__main__':
    unittest.main()

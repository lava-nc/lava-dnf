# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess

from lib.dnf.connect.connect import connect, validate_ops
from lib.dnf.connect.exceptions import MissingOpError, DuplicateOpError
from lib.dnf.operations.operations import AbstractOperation


class MockProcess(AbstractProcess):
    """Mock Process with an InPort and OutPort"""
    def __init__(self, shape):
        super().__init__()
        self.a_in = InPort(shape)
        self.s_out = OutPort(shape)


class MockOperation(AbstractOperation):
    """Mock Operation that generates an identity matrix as weights"""
    def __init__(self, shape=(1, 1)):
        super().__init__()
        self.shape = shape

    def compute_weights(self):
        return np.eye(self.shape[0], self.shape[1], dtype=np.int32)


class TestConnect(unittest.TestCase):
    def test_connect_function_exists_and_is_callable(self):
        """Tests whether the connect function exists and is callable."""
        import lib
        self.assertTrue(callable(getattr(lib.dnf.connect.connect, 'connect')))

    def test_connecting_source_and_target(self):
        """Tests connecting a source Process to a target Process."""
        # create mock processes and an operation to connect
        source = MockProcess(shape=(1, 2, 3))
        target = MockProcess(shape=(1, 2, 3))
        op = MockOperation(shape=(6, 6))

        # connect source to target
        connections = connect(source.s_out, target.a_in, ops=[op])

        # check whether the connect function returns a process
        self.assertIsInstance(connections, AbstractProcess)

        # check whether 'source' is connected to 'connections'
        src_op = source.out_ports.s_out
        con_ip = connections.in_ports.s_in
        self.assertEqual(src_op.get_dst_ports(), [con_ip])
        self.assertEqual(con_ip.get_src_ports(), [src_op])

        # check whether 'connections' is connected to 'target'
        con_op = connections.out_ports.a_out
        trg_op = target.in_ports.a_in
        self.assertEqual(con_op.get_dst_ports(), [trg_op])
        self.assertEqual(trg_op.get_src_ports(), [con_op])


class TestValidateOps(unittest.TestCase):
    def test_empty_operations_list_raises_value_error(self):
        """Tests whether an empty <ops> argument raises a value error."""
        with self.assertRaises(ValueError):
            validate_ops(ops=[], src_shape=(1,), dst_shape=(1,))

    def test_ops_list_containing_invalid_type_raises_type_error(self):
        """Tests whether the type of all elements in <ops> is validated."""
        class NotAnOperation:
            pass

        with self.assertRaises(TypeError):
            validate_ops(ops=[MockOperation(), NotAnOperation()],
                         src_shape=(1,),
                         dst_shape=(1,))

    def test_single_operation_can_be_passed_in_as_ops(self):
        """Tests whether a single operation is wrapped into a list."""
        ops = validate_ops(ops=MockOperation(shape=(1,)),
                           src_shape=(1,),
                           dst_shape=(1,))
        self.assertIsInstance(ops, list)
        self.assertIsInstance(ops[0], MockOperation)

    def test_duplicate_operations_changing_dim_raises_error(self):
        """Tests whether an exception is raised when the user specifies
        multiple operations that change the dimensionality."""

        class MockProjection(MockOperation):
            """Mock Projection operation"""
            def __init__(self, shape=(1, 1)):
                super().__init__(shape)
                self.changes_dim = True

        with self.assertRaises(DuplicateOpError) as context:
            validate_ops(src_shape=(2, 4),
                         dst_shape=(2,),
                         ops=[MockProjection(), MockProjection()])
        self.assertEqual(context.exception.duplicate_op, "changes_dim=True")

    def test_duplicate_operations_changing_size_raises_error(self):
        """Tests whether an exception is raised when the user specifies
        multiple operations that change the size."""

        class MockResize(MockOperation):
            """Mock Resize operation"""
            def __init__(self, shape=(1, 1)):
                super().__init__(shape)
                self.changes_size = True

        with self.assertRaises(DuplicateOpError) as context:
            validate_ops(src_shape=(5,),
                         dst_shape=(6,),
                         ops=[MockResize(), MockResize()])
        self.assertEqual(context.exception.duplicate_op, "changes_size=True")

    def test_duplicate_operations_reordering_shape_raises_error(self):
        """Tests whether an exception is raised when the user specifies
        multiple operations that reorder the shape."""

        class MockReorder(MockOperation):
            """Mock Reorder operation"""
            def __init__(self, shape=(1, 1)):
                super().__init__(shape)
                self.reorders_shape = True

        with self.assertRaises(DuplicateOpError) as context:
            validate_ops(src_shape=(5, 3),
                         dst_shape=(3, 5),
                         ops=[MockReorder(), MockReorder()])
        self.assertEqual(context.exception.duplicate_op, "reorders_shape=True")

    def test_shape_mismatch_with_correct_operation(self):
        """Tests whether validation passes when the shape of the source
        OutPort does not match the destination InPort and a Projection
        operation is specified."""

        class MockProjection(MockOperation):
            """Mock Projection operation"""
            def __init__(self, shape=(1, 1)):
                super().__init__(shape)
                self.changes_dim = True

        validate_ops(src_shape=(2, 4),
                     dst_shape=(2,),
                     ops=[MockProjection()])

    def test_shape_mismatch_without_correct_operation_raises_error(self):
        """Tests whether an error is raised when the shape of the source
        OutPort does not match the destination InPort and a Projection is
        required but missing."""
        with self.assertRaises(MissingOpError) as context:
            validate_ops(src_shape=(2, 4),
                         dst_shape=(2,),
                         ops=[MockOperation()])
        self.assertEqual(context.exception.missing_op, "changes_dim=True")

    def test_size_mismatch_with_correct_operation(self):
        """Tests whether validation passes when the sizes of the source
        OutPort do not match the destination InPort and a Resize
        operation is specified."""

        class MockResize(MockOperation):
            """Mock Resize operation"""
            def __init__(self, shape=(1, 1)):
                super().__init__(shape)
                self.changes_size = True

        validate_ops(src_shape=(5,),
                     dst_shape=(6,),
                     ops=[MockResize()])

    def test_size_mismatch_without_correct_operation_raises_error(self):
        """Tests whether an error is raised when the shape of the source
        OutPort does not match the destination InPort and a Resize is
        required but missing."""
        with self.assertRaises(MissingOpError) as context:
            validate_ops(src_shape=(5,),
                         dst_shape=(6,),
                         ops=[MockOperation()])
        self.assertEqual(context.exception.missing_op, "changes_size=True")

    def test_shape_order_mismatch_with_correct_operation(self):
        """Tests whether validation passes when the shape-order of the source
        OutPort does not match the destination InPort and a Reorder
        operation is specified."""

        class MockReorder(MockOperation):
            """Mock Reorder operation"""
            def __init__(self, shape=(1, 1)):
                super().__init__(shape)
                self.reorders_shape = True

        validate_ops(src_shape=(5, 3),
                     dst_shape=(3, 5),
                     ops=[MockReorder()])

    def test_shape_order_mismatch_without_correct_operation_raises_error(self):
        """Tests whether an error is raised when the shape of the source
        OutPort does not match the destination InPort and a Reorder is
        required but missing."""
        with self.assertRaises(MissingOpError) as context:
            validate_ops(src_shape=(5, 3),
                         dst_shape=(3, 5),
                         ops=[MockOperation()])
        self.assertEqual(context.exception.missing_op, "reorders_shape=True")


if __name__ == '__main__':
    unittest.main()

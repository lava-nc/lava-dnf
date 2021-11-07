# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess

from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.connect.exceptions import MisconfiguredConnectError
from lava.lib.dnf.operations.operations import AbstractComputedShapeOperation, \
    AbstractSpecifiedShapeOperation
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError
from lava.lib.dnf.utils.convenience import num_neurons


class MockProcess(AbstractProcess):
    """Mock Process with an InPort and OutPort"""
    def __init__(self, shape=(1,)):
        super().__init__()
        self.a_in = InPort(shape)
        self.s_out = OutPort(shape)


class MockNoChangeOperation(AbstractComputedShapeOperation):
    """Mock Operation that does not change shape"""
    def _compute_weights(self):
        return np.eye(num_neurons(self._output_shape),
                      num_neurons(self._input_shape),
                      dtype=np.int32)

    def _compute_output_shape(self, input_shape: ty.Tuple[int, ...]):
        self._output_shape = input_shape


class MockChangeOperation(AbstractSpecifiedShapeOperation):
    """Mock Operation that changes shape"""
    def __init__(self, output_shape):
        super().__init__(output_shape)

    def _compute_weights(self):
        return np.eye(num_neurons(self._output_shape),
                      num_neurons(self._input_shape),
                      dtype=np.int32)

    def _validate_output_shape(self):
        if self._input_shape == self._output_shape:
            raise MisconfiguredOpError("operation is intended to change shape")


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
        op = MockNoChangeOperation()

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
                    ops=[MockNoChangeOperation(), NotAnOperation()])

    def test_single_operation_is_automatically_wrapped_into_list(self):
        """Tests whether a single operation is wrapped into a list."""
        connect(MockProcess().s_out,
                MockProcess().a_in,
                ops=MockNoChangeOperation())

    def test_operation_that_changes_the_shape(self):
        """Tests whether an operation that changes shape can be used to
        connect two Processes."""
        output_shape = (3,)
        connect(MockProcess((5, 3)).s_out,
                MockProcess(output_shape).a_in,
                ops=MockChangeOperation(output_shape=output_shape))

    def test_mismatching_op_output_shape_and_dest_shape_raises_error(self):
        """Tests whether an error is raised when the output shape of the
        last operation does not match the destination shape."""
        with self.assertRaises(MisconfiguredConnectError):
            connect(MockProcess((5, 3)).s_out,
                    MockProcess((5,)).a_in,
                    ops=[MockNoChangeOperation()])

    def test_combining_multiple_ops_that_do_not_change_shape(self):
        """Tests whether multiple operations can be specified that do not
        change the shape."""
        connect(MockProcess((5, 3)).s_out,
                MockProcess((5, 3)).a_in,
                ops=[MockNoChangeOperation(),
                     MockNoChangeOperation()])

    def test_multiple_non_changing_ops_and_one_that_changes_shape(self):
        """Tests whether an operation that changes the shape
        can be combined with multiple operations that do not change the
        shape."""
        output_shape = (5,)
        connect(MockProcess((5, 3)).s_out,
                MockProcess(output_shape).a_in,
                ops=[MockNoChangeOperation(),
                     MockChangeOperation(output_shape),
                     MockNoChangeOperation()])

    def test_multiple_ops_that_change_shape(self):
        """Tests whether multiple operations that change shape can be
        combined with one that does not change shape."""
        connect(MockProcess((5, 3)).s_out,
                MockProcess((2,)).a_in,
                ops=[MockNoChangeOperation(),
                     MockChangeOperation(output_shape=(5, 2)),
                     MockChangeOperation(output_shape=(2,)),
                     MockNoChangeOperation()])

    def test_weights_from_multiple_ops_get_multiplied(self):
        """Tests whether compute_weights() multiplies the weights that are
        produced by all specified operations."""

        class MockNoChangeOpWeights(MockNoChangeOperation):
            """Mock Operation that generates an identity matrix with a given
            weight."""
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def _compute_weights(self):
                return np.eye(num_neurons(self._output_shape),
                              num_neurons(self._input_shape),
                              dtype=np.int32) * self.weight

        shape = (5, 3)
        w1 = 2
        w2 = 4

        conn = connect(MockProcess(shape).s_out,
                       MockProcess(shape).a_in,
                       ops=[MockNoChangeOpWeights(weight=w1),
                            MockNoChangeOpWeights(weight=w2)])

        computed_weights = conn.weights.get()
        expected_weights = np.eye(num_neurons(shape),
                                  num_neurons(shape),
                                  dtype=np.int32) * w1 * w2

        self.assertTrue(np.array_equal(computed_weights, expected_weights))


if __name__ == '__main__':
    unittest.main()

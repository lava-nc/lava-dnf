# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.population.process import Population
from lava.lib.dnf.operations.operations import Weights, ReduceDims, Reorder, \
    ExpandDims


class TestConnectingWithOperations(unittest.TestCase):
    def test_connect_population_with_weights_op(self):
        """Tests whether populations can be connected using the Weights
        operation."""
        for shape in [(1,), (5,), (5, 5), (5, 5, 5)]:
            source = Population(shape=shape)
            destination = Population(shape=shape)
            weights = Weights(5.0)
            connect(source.s_out, destination.a_in, ops=[weights])

    def test_connect_population_3d_to_2d_with_reduce_dims_and_reorder(self):
        """Tests whether reducing dimensions together with reordering works
        when going from 3D to 2D."""
        reduce_dims = [(2,), (1,), (2,), (1,), (0,), (0,)]
        orders = [(0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (1, 0)]

        matrices = [np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 1]]),
                    np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1, 0, 1]]),
                    np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 0, 0],
                              [0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 1]]),
                    np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 1, 0],
                              [0, 1, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 1]]),
                    np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1]]),
                    np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1]])]

        for dims, order, expected in zip(reduce_dims, orders, matrices):
            source = Population(shape=(2, 2, 2))
            destination = Population(shape=(2, 2))
            reorder_op = Reorder(order=order)
            reduce_op = ReduceDims(reduce_dims=dims)
            computed = connect(source.s_out,
                               destination.a_in,
                               ops=[reduce_op, reorder_op])

            self.assertTrue(np.array_equal(computed.weights.get(), expected))

    def test_connect_population_2d_to_3d_with_expand_dims_and_reorder(self):
        """Tests whether expanding dimensions together with reordering works
        when going from 2D to 3D."""
        orders = [(0, 1, 2),
                  (0, 2, 1),
                  (1, 0, 2),
                  (1, 2, 0),
                  (2, 0, 1),
                  (2, 1, 0)]

        matrices = [np.array([[1, 0, 0, 0],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1]]),
                    np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]),
                    np.array([[1, 0, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1]]),
                    np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]),
                    np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]]),
                    np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]])]

        for order, expected in zip(orders, matrices):
            source = Population(shape=(2, 2))
            destination = Population(shape=(2, 2, 2))
            reorder_op = Reorder(order=order)
            expand_op = ExpandDims(new_dims_shape=(2,))
            computed = connect(source.s_out,
                               destination.a_in,
                               ops=[expand_op, reorder_op])

            self.assertTrue(np.array_equal(computed.weights.get(), expected))

    def test_connect_population_1d_to_3d_with_expand_dims_and_reorder(self):
        """Tests whether expanding dimensions together with reordering works
        when going from 1D to 3D."""
        orders = [(0, 1, 2),
                  (1, 0, 2),
                  (2, 1, 0),
                  (0, 2, 1),
                  (2, 0, 1),
                  (1, 2, 0)]

        matrices = [np.array([[1, 0],
                              [1, 0],
                              [1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1],
                              [0, 1],
                              [0, 1]]),
                    np.array([[1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1],
                              [1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1]]),
                    np.array([[1, 0],
                              [0, 1],
                              [1, 0],
                              [0, 1],
                              [1, 0],
                              [0, 1],
                              [1, 0],
                              [0, 1]]),
                    np.array([[1, 0],
                              [1, 0],
                              [1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1],
                              [0, 1],
                              [0, 1]]),
                    np.array([[1, 0],
                              [0, 1],
                              [1, 0],
                              [0, 1],
                              [1, 0],
                              [0, 1],
                              [1, 0],
                              [0, 1]]),
                    np.array([[1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1],
                              [1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1]])]

        for order, expected in zip(orders, matrices):
            source = Population(shape=(2,))
            destination = Population(shape=(2, 2, 2))
            reorder_op = Reorder(order=order)
            expand_op = ExpandDims(new_dims_shape=(2, 2))
            computed = connect(source.s_out,
                               destination.a_in,
                               ops=[expand_op, reorder_op])

            self.assertTrue(np.array_equal(computed.weights.get(), expected))


if __name__ == '__main__':
    unittest.main()

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty
import numpy as np

from lava.lib.dnf.operations.operations import AbstractOperation, Weights
from lava.lib.dnf.operations.exceptions import MisconfiguredOpError
from lava.lib.dnf.utils.convenience import num_neurons


class MockOperation(AbstractOperation):
    def _compute_weights(self) -> np.ndarray:
        return np.ones((1, 1), dtype=np.int32)

    def _validate_configuration(self):
        pass


class TestAbstractOperation(unittest.TestCase):
    def test_computing_conn_without_prior_configuration_raises_error(self):
        """Tests whether an error is raised when compute_weights() is called
        before an operation has been configured."""
        op = MockOperation()
        with self.assertRaises(AssertionError):
            op.compute_weights()

    def test_computing_conn_with_prior_configuration_works(self):
        """Tests whether compute_weights() works and can be called once
        configuration is complete."""
        op = MockOperation()
        op.configure(input_shape=(1,), output_shape=(1,))
        computed_weights = op.compute_weights()
        expected_weights = np.ones((1, 1), dtype=np.int32)

        self.assertEqual(computed_weights, expected_weights)


if __name__ == '__main__':
    unittest.main()

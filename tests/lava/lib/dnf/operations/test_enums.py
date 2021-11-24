# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.dnf.operations.enums import ReduceMethod, BorderType


class TestReduceMethod(unittest.TestCase):
    def test_validate_sum(self) -> None:
        """Tests whether SUM is a valid type of the ReduceMethod enum."""
        ReduceMethod.validate(ReduceMethod.SUM)

    def test_validate_mean(self) -> None:
        """Tests whether MEAN is a valid type of the ReduceMethod enum."""
        ReduceMethod.validate(ReduceMethod.MEAN)

    def test_invalid_type_raises_type_error(self) -> None:
        """Tests whether int is an invalid type of the ReduceMethod enum."""
        with self.assertRaises(TypeError):
            ReduceMethod.validate(int)

    def test_invalid_value_raises_value_error(self) -> None:
        """Tests whether FOO is an invalid value of the ReduceMethod enum."""
        with self.assertRaises(AttributeError):
            _ = ReduceMethod.FOO


class TestBorderType(unittest.TestCase):
    def test_validate_padded(self) -> None:
        """Tests whether PADDED is a valid type of the BorderType enum."""
        BorderType.validate(BorderType.PADDED)

    def test_validate_circular(self) -> None:
        """Tests whether CIRCULAR is a valid type of the BorderType enum."""
        BorderType.validate(BorderType.CIRCULAR)

    def test_invalid_type_raises_type_error(self) -> None:
        """Tests whether int is an invalid type of the BorderType enum."""
        with self.assertRaises(TypeError):
            BorderType.validate(int)

    def test_invalid_value_raises_value_error(self) -> None:
        """Tests whether FOO is an invalid value of the BorderType enum."""
        with self.assertRaises(AttributeError):
            _ = BorderType.FOO

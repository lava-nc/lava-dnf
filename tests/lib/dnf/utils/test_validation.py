# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lib.dnf.utils.validation import validate_shape


class TestValidateShape(unittest.TestCase):
    def test_shape_int(self):
        """Tests whether the shape argument is converted to a tuple."""
        shape = validate_shape(shape=5)
        self.assertEqual(shape, (5,))

    def test_shape_tuple(self):
        """Tests whether a tuple shape argument remains a tuple."""
        shape = validate_shape(shape=(5, 3))
        self.assertTrue(shape == (5, 3))

    def test_shape_list(self):
        """Tests whether a list shape argument is converted to a tuple."""
        shape = validate_shape(shape=[5, 3])
        self.assertTrue(shape == (5, 3))

    def test_negative_values(self):
        """Tests whether negative shape values raise a ValueError."""
        with self.assertRaises(ValueError):
            validate_shape(shape=(5, -3))

    def test_invalid_type(self):
        """Tests whether an invalid type raises a TypeError."""
        with self.assertRaises(TypeError):
            validate_shape(shape=5.3)


if __name__ == '__main__':
    unittest.main()

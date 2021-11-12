# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.dnf.utils.math import is_odd


class TestIsOdd(unittest.TestCase):
    def test_is_odd(self):
        """Tests the is_odd() helper function."""
        self.assertFalse(is_odd(0))
        self.assertTrue(is_odd(1))


if __name__ == '__main__':
    unittest.main()

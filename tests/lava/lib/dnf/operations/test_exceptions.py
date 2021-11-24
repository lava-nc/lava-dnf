# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.dnf.operations.exceptions import MisconfiguredOpError


class TestMisconfiguredOpError(unittest.TestCase):
    def test_raising_misconfigured_op_error(self) -> None:
        """Tests whether the MisconfiguredOpError can be raised."""
        msg = "test message"
        with self.assertRaises(MisconfiguredOpError) as context:
            raise MisconfiguredOpError(msg)

        # check whether the message is set
        self.assertEqual(context.exception.args[0], msg)

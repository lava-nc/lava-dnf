# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.dnf.connect.exceptions import MisconfiguredConnectError


class TestMisconfiguredConnectError(unittest.TestCase):
    def test_raising_misconfigured_connect_error(self) -> None:
        """Tests whether the MisconfiguredConnectError can be raised."""
        msg = "test message"
        with self.assertRaises(MisconfiguredConnectError) as context:
            raise MisconfiguredConnectError(msg)

        # check whether the message is set
        self.assertEqual(context.exception.args[0], msg)


if __name__ == '__main__':
    unittest.main()

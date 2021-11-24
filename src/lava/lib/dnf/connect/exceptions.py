# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty


class MisconfiguredConnectError(Exception):
    """
    Exception that is raised when the connection function is misconfigured
    with a wrong combination of operations.

    Parameters:
    -----------
    msg : str (optional)
        custom exception message that overwrites the default
    """
    def __init__(self, msg: ty.Optional[str] = None) -> None:
        if msg is None:
            msg = "call to connection() misconfigured; check the choice and " \
                  "parameterization of all operations"
        super().__init__(msg)

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty


class MisconfiguredOpError(Exception):
    """
    Exception that is raised when an operation is misconfigured.

    Parameters:
    -----------
    msg : str (optional)
        custom exception message that overwrites the default
    """
    def __init__(self, msg: ty.Optional[str] = None) -> None:
        if msg is None:
            msg = "operation is misconfigured"
        super().__init__(msg)

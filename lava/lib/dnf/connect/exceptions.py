# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty


class MissingOpError(Exception):
    """
    Exception that is raised when a connection requires a particular
    operation but the user has not specified it.

    Parameters:
    -----------
    missing_op : str
        string representation of the missing operation
    msg : str
        custom exception message that overwrites the default
    """
    def __init__(self, missing_op: str, msg: ty.Optional[str] = None):
        if msg is None:
            msg = f"operation '{missing_op}' is required but missing"
        super().__init__(msg)
        self.missing_op = missing_op


class DuplicateOpError(Exception):
    """
    Exception that is raised when a user specifies a particular
    operation more than once in a list of <ops>.

    Parameters:
    -----------
    duplicate_op : str
        string representation of the duplicate operation
    msg : str
        custom exception message that overwrites the default
    """
    def __init__(self, duplicate_op: str, msg: ty.Optional[str] = None):
        if msg is None:
            msg = f"operation '{duplicate_op}' cannot be used more than once"
        super().__init__(msg)
        self.duplicate_op = duplicate_op

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


def is_odd(n):
    """
    Checks whether n is an odd number.

    :param int n: number to check
    :returns bool: True if <n> is an odd number"""
    return n & 1

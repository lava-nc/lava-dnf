# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np


def validate_shape(shape):
    """
    Validate and potentially convert shape parameter.

    The shape of different elements of the DNF library can be passed in as
    type tuple(int) or list(int) for multiple dimensions, or type int for a
    single dimension. In all cases, it is converted to tuple(int).

    Parameters:
    -----------
    shape : tuple(int) or list(int)
        shape parameter to be validated

    Returns:
    --------
    shape : tuple(int)
        validated and converted shape parameter
    """
    if shape is None:
        raise AssertionError("<shape> may not be None")

    # convert single int values to a tuple
    if isinstance(shape, int):
        shape = (shape,)
    # check whether all elements in the tuple (or list) are of type int
    # and positive
    if isinstance(shape, tuple) or isinstance(shape, list):
        for s in shape:
            if not isinstance(s, (int, np.integer)):
                raise TypeError("all elements of <shape> must be of type int")
            if s < 0:
                raise ValueError("all elements of <shape> must be greater "
                                 "than zero")
        # convert list to tuple
        if isinstance(shape, list):
            shape = tuple(shape)
    # if <shape> is not yet a tuple, raise a TypeError
    if not isinstance(shape, tuple):
        raise TypeError("<shape> must be of type int or tuple(int)")

    return shape

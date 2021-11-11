# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from enum import Enum, unique, auto


@unique
class ReduceMethod(Enum):
    """Enum for reduce methods of ReduceDims operation"""
    SUM = auto()  # ReduceDims will sum all synaptic weights of collapsed dim
    MEAN = auto()  # ReduceDims will compute mean of weights of collapsed dim

    @classmethod
    def validate(cls, reduce_method):
        """Validate type of <reduce_op>"""
        if not isinstance(reduce_method, ReduceMethod):
            raise TypeError("reduce_method must be of value ReduceMethod")

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from __future__ import annotations
from typing import TypeVar, Type
from enum import Enum, unique, auto

_T = TypeVar("_T")


@unique
class ReduceMethod(Enum):
    """Enum for reduce methods of ReduceDims operation"""
    SUM = auto()  # ReduceDims will sum all synaptic weights of collapsed dim
    MEAN = auto()  # ReduceDims will compute mean of weights of collapsed dim

    @classmethod
    def validate(cls: Type[_T], reduce_method: ReduceMethod) -> None:
        """Validate type of <reduce_op>"""
        if not isinstance(reduce_method, ReduceMethod):
            raise TypeError("reduce_method must be of type ReduceMethod")


@unique
class BorderType(Enum):
    PADDED = auto()
    CIRCULAR = auto()

    @classmethod
    def validate(cls: Type[_T], border_type: BorderType) -> None:
        """Validate type of <border_type>"""
        if not isinstance(border_type, BorderType):
            raise TypeError("border_type must be of type BorderType")

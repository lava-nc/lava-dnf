# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod


class AbstractOperation(ABC):
    def __init__(self):
        self.changes_dim = False
        self.changes_size = False
        self.reorders_shape = False

    @abstractmethod
    def compute_weights(self):
        pass

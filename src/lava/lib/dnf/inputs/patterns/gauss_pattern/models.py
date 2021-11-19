# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.lib.dnf.inputs.patterns.gauss_pattern.process import GaussPattern
from lava.lib.dnf.utils.math import gauss


@implements(proc=GaussPattern, protocol=LoihiProtocol)
@requires(CPU)
class GaussPatternProcessModel(PyLoihiProcessModel):
    _shape: np.ndarray = LavaPyType(np.ndarray, int)

    _amplitude: np.ndarray = LavaPyType(np.ndarray, float)
    _mean: np.ndarray = LavaPyType(np.ndarray, float)
    _stddev: np.ndarray = LavaPyType(np.ndarray, float)

    pattern: np.ndarray = LavaPyType(np.ndarray, float)

    changed: np.ndarray = LavaPyType(np.ndarray, bool)

    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        if self.changed[0]:
            self.pattern = gauss(shape=self._shape,
                                 domain=None,
                                 amplitude=self._amplitude[0],
                                 mean=self._mean,
                                 stddev=self._stddev)
            self.changed[0] = False
            self.a_out.send(self.pattern)

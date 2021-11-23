# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.utils.math import gauss


# TODO: (GK) Change protocol to AsyncProtocol when supported
# TODO: (GK) Change base class to (Sequential)PyProcessModel when supported
@implements(proc=GaussPattern, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class GaussPatternProcessModel(PyLoihiProcessModel):
    """
    PyLoihiProcessModel for GaussPatternProcess.

    Implements the behavior of sending a gauss pattern asynchronously when
    a change is triggered.
    """
    _shape: np.ndarray = LavaPyType(np.ndarray, int)

    _amplitude: np.ndarray = LavaPyType(np.ndarray, float)
    _mean: np.ndarray = LavaPyType(np.ndarray, float)
    _stddev: np.ndarray = LavaPyType(np.ndarray, float)

    null_pattern: np.ndarray = LavaPyType(np.ndarray, float)
    pattern: np.ndarray = LavaPyType(np.ndarray, float)

    changed: np.ndarray = LavaPyType(np.ndarray, bool)

    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        # When changed flag is set to True
        if self.changed[0]:
            # Compute new pattern based on updated parameters
            self.pattern = gauss(shape=self._shape,
                                 domain=None,
                                 amplitude=self._amplitude[0],
                                 mean=self._mean,
                                 stddev=self._stddev)
            # Reset changed flag
            self.changed[0] = False
            # Send new pattern through the PyOutPort
            self.a_out.send(self.pattern)
        else:
            # Send the null pattern
            self.a_out.send(self.null_pattern)

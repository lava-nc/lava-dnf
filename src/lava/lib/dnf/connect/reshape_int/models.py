# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU

from lava.lib.dnf.connect.reshape_int.process import ReshapeInt


@implements(proc=ReshapeInt, protocol=LoihiProtocol)
@requires(CPU)
class ReshapeIntProcessModel(PyLoihiProcessModel):
    """ProcessModel for the Reshape Process"""

    shape_out: np.ndarray = LavaPyType(np.ndarray, int)

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self) -> None:
        rec = self.s_in.recv()
        reshaped_input = np.reshape(rec, tuple(self.shape_out))
        self.s_out.send(reshaped_input)

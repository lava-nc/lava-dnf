# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class ProcessIn(AbstractProcess):
    def __init__(self,
                 user_input_shape: ty.Tuple[int, ...],
                 recv_pipe: mp.Pipe) -> None:
        super().__init__(user_input_shape=user_input_shape,
                         recv_pipe=recv_pipe)
        self.user_input_out_port = OutPort(shape=user_input_shape)


@implements(proc=ProcessIn, protocol=LoihiProtocol)
@requires(CPU)
class ProcessInPM(PyLoihiProcessModel):
    user_input_out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._user_input_shape = proc_params["user_input_shape"]
        self._recv_pipe = proc_params["recv_pipe"]

    def run_spk(self):
        data = np.zeros(self._user_input_shape, dtype=np.int32)

        if self._recv_pipe.poll():
            idx = self._recv_pipe.recv()
            data[idx] = 1

        self.user_input_out_port.send(data)

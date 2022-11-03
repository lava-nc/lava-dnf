# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class ProcessOut(AbstractProcess):
    def __init__(self,
                 dnf_shape: ty.Tuple[int, ...],
                 node_shape: ty.Tuple[int, ...],
                 send_pipe: mp.Pipe):
        super().__init__(dnf_shape=dnf_shape,
                         node_shape=node_shape,
                         send_pipe=send_pipe)

        self.inp_in = InPort(shape=dnf_shape)

        self.beh0_int_in = InPort(shape=node_shape)
        self.beh0_cos_in = InPort(shape=node_shape)

        self.beh1_int_in = InPort(shape=node_shape)
        self.beh1_cos_in = InPort(shape=node_shape)

        self.beh2_int_in = InPort(shape=node_shape)
        self.beh2_cos_in = InPort(shape=node_shape)

        self.beh0_beh1_pc_in = InPort(shape=node_shape)
        self.beh1_beh2_pc_in = InPort(shape=node_shape)

        self.wm0_in = InPort(shape=dnf_shape)
        self.wm1_in = InPort(shape=dnf_shape)
        self.wm2_in = InPort(shape=dnf_shape)

        self.gate0_in = InPort(shape=dnf_shape)
        self.gate1_in = InPort(shape=dnf_shape)
        self.gate2_in = InPort(shape=dnf_shape)

        self.cos_in = InPort(shape=dnf_shape)
        self.cod_in = InPort(shape=dnf_shape)

        self.cod_node_in = InPort(shape=node_shape)


@implements(proc=ProcessOut, protocol=LoihiProtocol)
@requires(CPU)
class ProcessOutPM(PyLoihiProcessModel):
    inp_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    beh0_int_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    beh0_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    beh1_int_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    beh1_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    beh2_int_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    beh2_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    beh0_beh1_pc_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    beh1_beh2_pc_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    wm0_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    wm1_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    wm2_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    gate0_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    gate1_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    gate2_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    cod_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    cod_node_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._dnf_shape = proc_params["dnf_shape"]
        self._node_shape = proc_params["node_shape"]
        self._send_pipe = proc_params["send_pipe"]

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        inp = self.inp_in.recv()

        beh0_int = self.beh0_int_in.recv()
        beh0_cos = self.beh0_cos_in.recv()

        beh1_int = self.beh1_int_in.recv()
        beh1_cos = self.beh1_cos_in.recv()

        beh2_int = self.beh2_int_in.recv()
        beh2_cos = self.beh2_cos_in.recv()

        beh0_beh1_pc = self.beh0_beh1_pc_in.recv()
        beh1_beh2_pc = self.beh1_beh2_pc_in.recv()

        wm0 = self.wm0_in.recv()
        wm1 = self.wm1_in.recv()
        wm2 = self.wm2_in.recv()

        gate0 = self.gate0_in.recv()
        gate1 = self.gate1_in.recv()
        gate2 = self.gate2_in.recv()

        cos = self.cos_in.recv()
        cod = self.cod_in.recv()

        cod_node = self.cod_node_in.recv()

        data_dict = {
            "inp_data": inp,
            "beh0_int_data": beh0_int,
            "beh0_cos_data": beh0_cos,
            "beh1_int_data": beh1_int,
            "beh1_cos_data": beh1_cos,
            "beh2_int_data": beh2_int,
            "beh2_cos_data": beh2_cos,
            "beh0_beh1_pc_data": beh0_beh1_pc,
            "beh1_beh2_pc_data": beh1_beh2_pc,
            "wm0_data": wm0,
            "wm1_data": wm1,
            "wm2_data": wm2,
            "gate0_data": gate0,
            "gate1_data": gate1,
            "gate2_data": gate2,
            "cos_data": cos,
            "cod_data": cod,
            "cod_node_data": cod_node
        }

        self._send_pipe.send(data_dict)

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
import cv2


class ProcessOut(AbstractProcess):
    """ Process that receives (1) the raw dvs events and (2,3) the spike rates
    of the selective as well as the multipeak dnf per pixel. It sends these
    values through a pipe to allow for plotting."
    """
    def __init__(self,
                 shape_dvs_frame,
                 shape_dnf,
                 send_pipe):
        super().__init__(shape_dvs_frame=shape_dvs_frame,
                         shape_dnf=shape_dnf,
                         send_pipe=send_pipe)
        self.dvs_frame_port = InPort(shape=shape_dvs_frame)
        self.dnf_multipeak_rates_port = InPort(shape=shape_dnf)
        self.dnf_selective_rates_port = InPort(shape=shape_dnf)


@implements(proc=ProcessOut, protocol=LoihiProtocol)
@requires(CPU)
class DataRelayerPM(PyLoihiProcessModel):
    # declare inports
    # Note: no outports needed since everything is sent through
    # predeclared pipe.
    dvs_frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    dnf_multipeak_rates_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    dnf_selective_rates_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._send_pipe = proc_params["send_pipe"]

    def run_spk(self):
        dvs_frame = self.dvs_frame_port.recv()
        dnf_multipeak_rates = self.dnf_multipeak_rates_port.recv()
        dnf_selective_rates = self.dnf_selective_rates_port.recv()

        # prepare frames for plotting
        dvs_frame_ds_image = cv2.rotate(dvs_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        dnf_multipeak_rates_ds_image = cv2.rotate(dnf_multipeak_rates, cv2.ROTATE_90_COUNTERCLOCKWISE)
        dnf_selective_rates_ds_image = cv2.rotate(dnf_selective_rates, cv2.ROTATE_90_COUNTERCLOCKWISE)

        data_dict = {
            "dvs_frame_ds_image": dvs_frame_ds_image,
            "dnf_multipeak_rates_ds_image": dnf_multipeak_rates_ds_image,
            "dnf_selective_rates_ds_image": dnf_selective_rates_ds_image,
        }
        # send data through pipe
        self._send_pipe.send(data_dict)

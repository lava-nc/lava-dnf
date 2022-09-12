import logging

from lava.proc.lif.process import LIF
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights
from lava.lib.dnf.operations.operations import Convolution
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
from lava.proc.monitor.models import PyMonitorModel

from utils import plot_1d, animated_1d_plot

import unittest
import numpy as np

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.c.model import CLoihiProcessModel
from lava.magma.core.model.c.ports import CInPort, COutPort
from lava.magma.core.model.c.type import LavaCType, LavaCDataType
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU, LMT
from lava.proc.conv.process import Conv
from tests.lava.test_utils.utils import Utils

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense

from dvs_file_input.process import DVSFileInput


class PyRecvVecDense(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop('shape')
        num_time_steps = kwargs.pop('num_time_steps')
        self.in_port = InPort(shape)
        self.var = Var(shape=(num_time_steps,) + shape)


@implements(proc=PyRecvVecDense, protocol=LoihiProtocol)
@requires(CPU)
class ProcModelPyRecvVecDense(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, 8)
    var: np.ndarray = LavaPyType(np.ndarray, np.int32)
    time_step: int = 0

    def spk_guard(self):
        return True

    def run_spk(self):
        print("run spk ProcModelPyRecvVecDense")
        self.var[self.time_step, :] = self.in_port.recv()
        print("after receive")
        self.time_step += 1


class Injector(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        self.in_port = InPort(shape)
        self.out_port = OutPort(shape)


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(LMT)
class InjectorModel(CLoihiProcessModel):
    in_port: CInPort = LavaCType(CInPort, LavaCDataType.INT32)
    out_port: COutPort = LavaCType(COutPort, LavaCDataType.INT32)

    @property
    def source_file_name(self) -> str:
        return "injector.c"


class SpikeReader(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        self.in_port = InPort(shape)
        self.out_port = OutPort(shape)


@implements(proc=SpikeReader, protocol=LoihiProtocol)
@requires(LMT)
class SpikeReaderModel(CLoihiProcessModel):
    in_port: CInPort = LavaCType(CInPort, LavaCDataType.INT32)
    out_port: COutPort = LavaCType(COutPort, LavaCDataType.INT32)

    @property
    def source_file_name(self) -> str:
        return "spikereader.c"


class Architecture:
    """This class structure is not required and is only used here to reduce
    code duplication for different examples."""

    def __init__(self, kernel):
        shape = (15,)
        time_steps = 700

        # Set up spike generator 1
        self.gauss_pattern_1 = GaussPattern(shape=shape,
                                            amplitude=0,
                                            mean=11.25,
                                            stddev=2.25)
        self.spike_generator_1 = RateCodeSpikeGen(shape=shape)
        self.injector_1 = Injector(shape=shape, log_config=LogConfig(
            level=logging.ERROR))
        self.gauss_pattern_1.a_out.connect(self.spike_generator_1.a_in)
        self.spike_generator_1.s_out.connect(self.injector_1.in_port)

        # Set up spike generator 2
        # self.gauss_pattern_2 = GaussPattern(shape=shape,
        #                                     amplitude=0,
        #                                     mean=3.75,
        #                                     stddev=2.25)
        # self.spike_generator_2 = RateCodeSpikeGen(shape=shape)
        # self.injector_2 = Injector(shape=shape, log_config=LogConfig(
        #     level=logging.ERROR))
        # self.gauss_pattern_2.a_out.connect(self.spike_generator_2.a_in)
        # self.spike_generator_2.s_out.connect(self.injector_2.in_port)

        # DNF with specified kernel
        self.dnf = LIF(shape=shape, du=409, dv=2047, vth=200)
        connect(self.dnf.s_out, self.dnf.a_in, [Convolution(kernel)])

        # Connect spike input to DNF
        connect(self.injector_1.out_port, self.dnf.a_in, [Weights(25)])
        # connect(self.injector_2.out_port, self.dnf.a_in, [Weights(25)])

        # Set up monitors
        self.spike_reader = SpikeReader(shape=shape, log_config=LogConfig(
            level=logging.ERROR))
        self.py_receiver = PyRecvVecDense(shape=shape,
                                          num_time_steps=time_steps)
        self.dnf.s_out.connect(self.spike_reader.in_port)
        self.spike_reader.out_port.connect(self.py_receiver.in_port)

        self.monitor_input_1 = Monitor()
        self.monitor_input_1.probe(self.spike_generator_1.s_out, time_steps)
        # self.monitor_input_2 = Monitor()
        # self.monitor_input_2.probe(self.spike_generator_2.s_out, time_steps)

        # Set up a run configuration
        self.run_cfg = Loihi2HwCfg()

    def run(self):
        # Run the network and make changes to spike inputs over time
        print("time step: 0")
        condition = RunSteps(num_steps=1)
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        print("time step: 1")
        condition = RunSteps(num_steps=99)
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 2300
        # self.gauss_pattern_2.amplitude = 2300
        print("time step: 100")
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 11200
        # self.gauss_pattern_2.amplitude = 11200
        print("time step: 200")
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 2300
        # self.gauss_pattern_2.amplitude = 2300
        print("time step: 300")
        self.gauss_pattern_1.run(condition=RunSteps(num_steps=200),
                                 run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 0
        # self.gauss_pattern_2.amplitude = 0
        print("time step: 500")
        self.gauss_pattern_1.run(condition=RunSteps(num_steps=200),
                                 run_cfg=self.run_cfg)
        print("time step: 700")

    def plot(self):
        # Get probed data from monitors
        data_dnf = self.py_receiver.var.get()
        data_input1 = self.monitor_input_1.get_data() \
            [self.spike_generator_1.name][self.spike_generator_1.s_out.name]
        # data_input2 = self.monitor_input_2.get_data() \
        #     [self.spike_generator_2.name][self.spike_generator_2.s_out.name]


        # Stop the execution of the network
        self.spike_generator_1.stop()

        # Generate a raster plot from the probed data
        plot_1d(data_dnf,
                data_input1, data_input1)#,
                #data_input2)

        # Generate an animated plot from the probed data
        # animated_1d_plot(data_dnf,
        #                 data_input1,
        #                 data_input2)


if __name__ == '__main__':
    from lava.lib.dnf.kernels.kernels import MultiPeakKernel

    detection_kernel = MultiPeakKernel(amp_exc=83,
                                       width_exc=3.75,
                                       amp_inh=-70,
                                       width_inh=7.5)

    architecture = Architecture(detection_kernel)
    architecture.run()
    architecture.plot()

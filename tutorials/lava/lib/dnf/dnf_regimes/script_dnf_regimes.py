import logging

from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
from lava.proc.lif.process import LIF
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights
from lava.lib.dnf.operations.operations import Convolution
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
from lava.proc.monitor.models import PyMonitorModel
from lava.lib.dnf.kernels.kernels import MultiPeakKernel, Kernel

from utils import plot_1d, animated_1d_plot

import unittest
import numpy as np
import typing as ty

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


class PyInjector(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop('shape')
        self.out_port = OutPort(shape)


@implements(proc=PyInjector, protocol=LoihiProtocol)
@requires(CPU)
class PyInjectorModel(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, 8)

    def spk_guard(self):
        return True

    def run_spk(self):
        print("run spk PyInjectorModel")
        self.out_port.send(data=np.ones(self.out_port.shape))



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

    def spk_guard(self):
        return True

    def run_spk(self):
        print("run spk ProcModelPyRecvVecDense")
        self.var[self.time_step - 1, :] = self.in_port.recv()
        print("after receive")


class Sum(AbstractProcess):
    def __init__(self, shape: ty.Tuple[int, ...]) -> None:
        super().__init__(shape=shape)

        self.in_port_1 = InPort(shape=shape)
        self.in_port_2 = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)


@implements(proc=Sum, protocol=LoihiProtocol)
@requires(CPU)
class SumProcessModel(PyLoihiProcessModel):
    in_port_1: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    in_port_2: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)

    # TODO (MR): Fix the bug (?) with multiple Dense Processes that have
    #  different num_message_bit values.
    def run_spk(self) -> None:
        input_1 = self.in_port_1.recv()
        input_2 = self.in_port_2.recv()
        #input_sum = input_1.astype(int) + input_2.astype(int)
        input_or = input_1 + input_2

        print(f"{input_or}")

        self.out_port.send(input_or)


class Architecture:
    """This class structure is not required and is only used here to reduce
    code duplication for different examples."""

    def __init__(self,
                 kernel: Kernel,
                 loihi2: bool) -> None:
        shape = (15,)
        time_steps = 700

        # Set up spike generator 1
        self.gauss_pattern_1 = GaussPattern(shape=shape,
                                            amplitude=0,
                                            mean=11.25,
                                            stddev=2.25)
        self.gauss_pattern_2 = GaussPattern(shape=shape,
                                            amplitude=0,
                                            mean=3.75,
                                            stddev=2.25)

        self.spike_generator_1 = RateCodeSpikeGen(shape=shape)
        self.spike_generator_2 = RateCodeSpikeGen(shape=shape)
        self.sum = Sum(shape=shape)
        if loihi2:
            self.injector = PyToNxAdapter(shape=shape)

        self.input_dense = Dense(weights=np.eye(shape[0]) * 25)

        self.dnf = LIF(shape=shape, du=409, dv=2047, vth=200)
        dense = connect(self.dnf.s_out, self.dnf.a_in, [Convolution(kernel)])

        print(f"{dense.weights.get()=}")

        # ops = [Convolution(kernel)]
        # _configure_ops(ops=[Convolution(kernel)],
        #                self.dnf.s_out.shape,
        #                self.dnf.a_in.shape)
        # weights = _compute_weights(ops)

        # self.dense_dnf = Dense(weights=np.full(shape=(shape[0], shape[0]),
        #                                        fill_value=254))
        # self.dnf.s_out.connect(self.dense_dnf.s_in)
        # self.dense_dnf.a_out.connect(self.dnf.a_in)

        # Set up monitors
        if loihi2:
            self.spike_reader = NxToPyAdapter(shape=shape)
        self.py_receiver = PyRecvVecDense(shape=shape,
                                          num_time_steps=time_steps)

        self.gauss_pattern_1.a_out.connect(self.spike_generator_1.a_in)
        self.gauss_pattern_2.a_out.connect(self.spike_generator_2.a_in)

        self.spike_generator_1.s_out.connect(self.sum.in_port_1)
        self.spike_generator_2.s_out.connect(self.sum.in_port_2)
        if loihi2:
            self.sum.out_port.connect(self.injector.inp)
            self.injector.out.connect(self.input_dense.s_in)
        else:
            self.sum.out_port.connect(self.input_dense.s_in)
            #self.spike_generator_1.s_out.connect(self.input_dense.s_in)
            #self.spike_generator_2.s_out.connect(self.input_dense.s_in)

        self.input_dense.a_out.connect(self.dnf.a_in)

        if loihi2:
            self.dnf.s_out.connect(self.spike_reader.inp)
            self.spike_reader.out.connect(self.py_receiver.in_port)
        else:
            self.dnf.s_out.connect(self.py_receiver.in_port)

        self.monitor_input_1 = Monitor()
        self.monitor_input_1.probe(self.spike_generator_1.s_out, time_steps)
        self.monitor_input_2 = Monitor()
        self.monitor_input_2.probe(self.spike_generator_2.s_out, time_steps)

        # Set up a run configuration
        if loihi2:
            self.run_cfg = Loihi2HwCfg()
        else:
            self.run_cfg = Loihi1SimCfg(select_tag="fixed_pt")

    def run(self):
        # Run the network and make changes to spike inputs over time
        print("time step: 0")
        condition = RunSteps(num_steps=100)
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 2300
        self.gauss_pattern_2.amplitude = 2300
        print("time step: 100")
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 11200
        self.gauss_pattern_2.amplitude = 11200
        print("time step: 200")
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 2300
        self.gauss_pattern_2.amplitude = 2300
        print("time step: 300")
        self.gauss_pattern_1.run(condition=RunSteps(num_steps=200),
                                 run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 0
        self.gauss_pattern_2.amplitude = 0
        print("time step: 500")
        self.gauss_pattern_1.run(condition=RunSteps(num_steps=200),
                                 run_cfg=self.run_cfg)
        print("time step: 700")

    def plot(self):
        # Get probed data from monitors
        data_dnf = self.py_receiver.var.get()
        data_input1 = self.monitor_input_1.get_data() \
            [self.spike_generator_1.name][self.spike_generator_1.s_out.name]
        data_input2 = self.monitor_input_2.get_data() \
            [self.spike_generator_2.name][self.spike_generator_2.s_out.name]

        # Stop the execution of the network
        self.dnf.stop()

        # Generate a raster plot from the probed data
        plot_1d(data_dnf,
                data_input1,
                data_input2)

        # Generate an animated plot from the probed data
        # animated_1d_plot(data_dnf,
        #                 data_input1,
        #                 data_input2)


if __name__ == '__main__':
    detection_kernel = MultiPeakKernel(amp_exc=83,
                                       width_exc=3.75,
                                       amp_inh=-70,
                                       width_inh=7.5)

    architecture = Architecture(detection_kernel, loihi2=True)
    architecture.run()
    architecture.plot()

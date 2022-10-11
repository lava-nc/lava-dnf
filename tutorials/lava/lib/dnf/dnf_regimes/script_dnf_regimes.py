import numpy as np
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
from lava.proc.io.sink import RingBuffer
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Convolution
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.monitor.process import Monitor
from lava.lib.dnf.kernels.kernels import MultiPeakKernel, Kernel
from utils import plot_1d, animated_1d_plot
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense


class Architecture:
    """This class structure is not required and is only used here to reduce
    code duplication for different examples."""

    def __init__(self,
                 kernel: Kernel,
                 loihi2: bool) -> None:
        shape = (15,)
        time_steps = 700

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

        if loihi2:
            self.injector1 = PyToNxAdapter(shape=shape)
            self.injector2 = PyToNxAdapter(shape=shape)

        self.input_dense1 = Dense(weights=np.eye(shape[0]) * 25)
        self.input_dense2 = Dense(weights=np.eye(shape[0]) * 25)

        self.dnf = LIF(shape=shape, du=409, dv=2047, vth=200)
        dense = connect(self.dnf.s_out, self.dnf.a_in, [Convolution(kernel)])

        if loihi2:
            self.spike_reader = NxToPyAdapter(shape=shape)
        self.py_receiver = RingBuffer(shape=shape, buffer=time_steps)

        self.gauss_pattern_1.a_out.connect(self.spike_generator_1.a_in)
        self.gauss_pattern_2.a_out.connect(self.spike_generator_2.a_in)

        if loihi2:
            self.spike_generator_1.s_out.connect(self.injector1.inp)
            self.spike_generator_2.s_out.connect(self.injector2.inp)
            self.injector1.out.connect(self.input_dense1.s_in)
            self.injector2.out.connect(self.input_dense2.s_in)
        else:
            self.spike_generator_1.s_out.connect(self.input_dense1.s_in)
            self.spike_generator_2.s_out.connect(self.input_dense2.s_in)

        self.input_dense1.a_out.connect(self.dnf.a_in)
        self.input_dense2.a_out.connect(self.dnf.a_in)

        if loihi2:
            self.dnf.s_out.connect(self.spike_reader.inp)
            self.spike_reader.out.connect(self.py_receiver.a_in)
        else:
            self.dnf.s_out.connect(self.py_receiver.a_in)

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
        condition = RunSteps(num_steps=100)
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 2300
        self.gauss_pattern_2.amplitude = 2300
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 11200
        self.gauss_pattern_2.amplitude = 11200
        self.gauss_pattern_1.run(condition=condition, run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 2300
        self.gauss_pattern_2.amplitude = 2300
        self.gauss_pattern_1.run(condition=RunSteps(num_steps=200),
                                 run_cfg=self.run_cfg)
        self.gauss_pattern_1.amplitude = 0
        self.gauss_pattern_2.amplitude = 0
        self.gauss_pattern_1.run(condition=RunSteps(num_steps=200),
                                 run_cfg=self.run_cfg)


    def plot(self):
        # Get probed data from monitors
        data_dnf = self.py_receiver.data.get().transpose()
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

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from multiprocessing import Pipe
from lava.lib.dnf.connect.connect import connect
from lava.magma.core.run_conditions import RunSteps
from demos.motion_tracking.dvs_file_input.process import DVSFileInput, PyDVSFileInputPM
from lava.proc.lif.process import LIF
from lava.proc.embedded_io.spike import NxToPyAdapter, PyToNxAdapter
from lava.lib.dnf.kernels.kernels import MultiPeakKernel, SelectiveKernel
from lava.lib.dnf.operations.operations import Convolution, Weights
from demos.motion_tracking.process_out.process import ProcessOut, ProcessOutModel
from lava.magma.core.run_configs import Loihi2HwCfg
from demos.motion_tracking.rate_reader.process import RateReader
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.runtime.runtime import Runtime
from lava.magma.compiler.executable import Executable
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions

CompilerOptions.verbose = True

# Default configs for the motion tracking network
dvs_file_input_default_config = {
    "true_height": 180,
    "true_width": 240,
    "file_path": "dvs_recording.aedat4",
    "flatten": False,
    "downsample_factor": 8,
    "downsample_mode": "max_pooling"
}

multipeak_dnf_default_config = {
    "in_conn": {
        "weight": 8
    },
    "lif": {"du": 2000,
            "dv": 2000,
            "vth": 30

            },
    "rec_conn": {
        "amp_exc": 14,
        "width_exc": [5, 5],
        "amp_inh": -10,
        "width_inh": [9, 9]
    },
    "out_conn": {
        "weight": 20,
    }
}

selective_dnf_default_config = {
    "lif": {"du": 809,
            "dv": 2047,
            "vth": 30
            },
    "rec_conn": {
        "amp_exc": 7,
        "width_exc": [7, 7],
        "global_inh": -5
    }
}

rate_reader_default_config = {
    "buffer_size": 10,
}


class MotionTracker:
    """Class to setup the motion tracking network, compile it, and initialize
     its runtime. The network topology looks as follows:

     DVS_Input -> PytoNxAdapter -> Multipeak DNF -> Selective DNF
         |                              |                |
         |                           NxtoPyAdapter    NxtoPyAdapter
         |                              |                |
         |                           RateReader       RateReader
         |                              |                 |
         ----------------------------> DataRelay <--------
    """

    def __init__(self,
                 send_pipe: type(Pipe),
                 num_steps: int,
                 blocking: ty.Optional[bool] = False,
                 dvs_file_input_config: ty.Optional[dict] = None,
                 multipeak_dnf_config: ty.Optional[dict] = None,
                 selective_dnf_config: ty.Optional[dict] = None,
                 rate_reader_config: ty.Optional[dict] = None,
                 executable: ty.Optional[Executable] = None) -> None:

        # Initialize input file/data
        dvs_file_input_config = \
            dvs_file_input_config or dvs_file_input_default_config
        multipeak_dnf_config = \
            multipeak_dnf_config or multipeak_dnf_default_config
        selective_dnf_config = \
            selective_dnf_config or selective_dnf_default_config
        rate_reader_config = \
            rate_reader_config or rate_reader_default_config

        # Initialize input params
        self.true_shape = (dvs_file_input_config["true_width"],
                           dvs_file_input_config["true_height"])
        self.file_path = dvs_file_input_config["file_path"]
        self.downsample_factor = dvs_file_input_config["downsample_factor"]
        self.downsample_mode = dvs_file_input_config["downsample_mode"]
        self.flatten = dvs_file_input_config["flatten"]
        self.downsampled_shape = (self.true_shape[0] // self.downsample_factor,
                                   self.true_shape[1] // self.downsample_factor)

        # Initialize multipeak dnf params
        self.multipeak_in_params = multipeak_dnf_config["in_conn"]
        self.multipeak_lif_params = multipeak_dnf_config["lif"]
        self.multipeak_rec_params = multipeak_dnf_config["rec_conn"]
        self.multipeak_out_params = multipeak_dnf_config["out_conn"]

        # Initialize selective dnf params
        self.selective_lif_params = selective_dnf_config["lif"]
        self.selective_rec_params = selective_dnf_config["rec_conn"]

        # Intialize rate reader params
        self.buffer_size_rate_reader = rate_reader_config["buffer_size"]

        # Initialize send_pipe
        self.send_pipe = send_pipe

        self._create_processes()
        self._make_connections()

        # Runtime Creation and Compilation
        exception_pm_map = {
            DVSFileInput: PyDVSFileInputPM,
            ProcessOut: ProcessOutModel
        }
        run_cfg = Loihi2HwCfg(exception_proc_model_map=exception_pm_map)
        self.num_steps = num_steps
        self.blocking = blocking

        # Compilation
        compiler = Compiler()

        if executable is None:
            executable = compiler.compile(self.dvs_file_input, run_cfg=run_cfg)

        # Initialize runtime
        mp = ActorType.MultiProcessing
        self.runtime = Runtime(exe=executable,
                               message_infrastructure_type=mp)
        self.runtime.initialize()

    def _create_processes(self) -> None:
        # Instantiate Processes Running on CPU
        self.dvs_file_input = \
            DVSFileInput(true_height=self.true_shape[1],
                         true_width=self.true_shape[0],
                         file_path=self.file_path,
                         flatten=self.flatten,
                         down_sample_factor=self.downsample_factor,
                         down_sample_mode=self.downsample_mode)

        self.rate_reader_multi_peak = \
            RateReader(shape=self.downsampled_shape,
                       buffer_size=self.buffer_size_rate_reader)

        self.rate_reader_selective = \
            RateReader(shape=self.downsampled_shape,
                       buffer_size=self.buffer_size_rate_reader)

        # sends data to pipe for plotting
        self.data_relayer = ProcessOut(shape_dvs_frame=self.downsampled_shape,
                                       shape_dnf=self.downsampled_shape,
                                       send_pipe=self.send_pipe)

        # Instantiate C-Processes Running on LMT
        self.c_injector = PyToNxAdapter(shape=self.downsampled_shape)
        self.c_spike_reader_multi_peak = NxToPyAdapter(
            shape=self.downsampled_shape)
        self.c_spike_reader_selective = NxToPyAdapter(
            shape=self.downsampled_shape)

        # Instantiate Processes Running on Loihi 2
        self.dnf_multipeak = LIF(shape=self.downsampled_shape,
                                 **self.multipeak_lif_params)
        self.dnf_selective = LIF(shape=self.downsampled_shape,
                                 **self.selective_lif_params)

    def _make_connections(self) -> None:
        # Connecting Input Processes
        self.dvs_file_input.event_frame_out.connect(self.c_injector.inp)

        # Connections around multipeak dnf
        connect(self.c_injector.out, self.dnf_multipeak.a_in,
                ops=[Weights(**self.multipeak_in_params)])
        connect(self.dnf_multipeak.s_out, self.dnf_multipeak.a_in,
                ops=[Convolution(MultiPeakKernel(**self.multipeak_rec_params))])
        connect(self.dnf_multipeak.s_out, self.dnf_selective.a_in,
                ops=[Weights(**self.multipeak_out_params)])

        # Connections around selective dnf
        connect(self.dnf_selective.s_out, self.dnf_selective.a_in,
                ops=[Convolution(SelectiveKernel(**self.selective_rec_params))])

        # Connect C Reader Processes
        self.dnf_multipeak.s_out.connect(
            self.c_spike_reader_multi_peak.inp)
        self.dnf_selective.s_out.connect(
            self.c_spike_reader_selective.inp)

        # Connect RateReaders
        self.c_spike_reader_multi_peak.out.connect(
            self.rate_reader_multi_peak.in_port)
        self.c_spike_reader_selective.out.connect(
            self.rate_reader_selective.in_port)

        # Connect ProcessOut (data relayer)
        self.dvs_file_input.event_frame_out.reshape(
            new_shape=self.downsampled_shape).connect(
            self.data_relayer.dvs_frame_port)
        self.rate_reader_multi_peak.out_port.connect(
            self.data_relayer.dnf_multipeak_rates_port)
        self.rate_reader_selective.out_port.connect(
            self.data_relayer.dnf_selective_rates_port)

    def start(self) -> None:
        self.runtime.start(RunSteps(num_steps=self.num_steps, blocking=self.blocking))

    def stop(self) -> None:
        self.runtime.wait()
        self.runtime.stop()

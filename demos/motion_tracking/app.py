# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from threading import Thread
from multiprocessing import Pipe
from functools import partial

from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, Spacer
from bokeh.models import LinearColorMapper, ColorBar, Title, Button
from bokeh.models.ranges import DataRange1d

from lava.proc.lif.process import LIF
from lava.proc.embedded_io.spike import NxToPyAdapter, PyToNxAdapter
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.runtime.runtime import Runtime
from lava.magma.core.run_conditions import RunSteps


from lava.lib.dnf.connect.connect import _configure_ops, _compute_weights
from lava.lib.dnf.kernels.kernels import MultiPeakKernel, SelectiveKernel
from lava.lib.dnf.operations.operations import Convolution
from lava.lib.dnf.demos.motion_tracking.sparse.process import Sparse

from process_out.process import ProcessOut, DataRelayerPM
from dvs_file_input.process import DVSFileInput, PyDVSFileInputPM
from rate_reader.process import RateReader


# ==========================================================================
# Parameters
# ==========================================================================
# number of time steps to be run in demo
num_steps = 4800

# DVSFileInput Params
true_height = 180
true_width = 240
file_path = "dvs_recording.aedat4"
flatten = True
down_sample_factor = 8
down_sample_mode = "max_pooling"

down_sampled_shape = (true_width // down_sample_factor,
                      true_height // down_sample_factor)

num_neurons = (true_height // down_sample_factor) * \
              (true_width // down_sample_factor)
down_sampled_flat_shape = (num_neurons,)

true_shape = (true_width, true_height)

# RateReader Params
buffer_size_rate_reader = 10

# Network Params
multipeak_params = {"amp_exc": 14,
                    "width_exc": [5, 5],
                    "amp_inh": -10,
                    "width_inh": [10, 10]}

selective_kernel_params = {"amp_exc": 7,
                           "width_exc": [7, 7],
                           "global_inh": -5}
sparse1_weights = 8
sparse2_weights = 20
dv_selective = 2047
du_selective = 809
selective_threshold = 30
du_multipeak = 2000
dv_multipeak = 2000
multipeak_threshold = 30

# MultiPeak DNF Params
kernel_multi_peak = MultiPeakKernel(**multipeak_params)
ops_multi_peak = [Convolution(kernel_multi_peak)]
_configure_ops(ops_multi_peak, down_sampled_shape, down_sampled_shape)
weights_multi_peak = _compute_weights(ops_multi_peak)

# Selective DNF Params
kernel_selective = SelectiveKernel(**selective_kernel_params)
ops_selective = [Convolution(kernel_selective)]
_configure_ops(ops_selective, down_sampled_shape, down_sampled_shape)
weights_selective = _compute_weights(ops_selective)

# ==========================================================================
# Instantiating Pipes
# ==========================================================================
recv_pipe, send_pipe = Pipe()

# ==========================================================================
# Instantiate Processes Running on CPU
# ==========================================================================
dvs_file_input = DVSFileInput(true_height=true_height,
                              true_width=true_width,
                              file_path=file_path,
                              flatten=flatten,
                              down_sample_factor=down_sample_factor,
                              down_sample_mode=down_sample_mode,
                              num_steps=num_steps)

rate_reader_multi_peak = RateReader(shape=down_sampled_shape,
                                    buffer_size=buffer_size_rate_reader,
                                    num_steps=num_steps)

rate_reader_selective = RateReader(shape=down_sampled_shape,
                                   buffer_size=buffer_size_rate_reader,
                                   num_steps=num_steps)

# sends data to pipe for plotting
data_relayer = ProcessOut(shape_dvs_frame=down_sampled_shape,
                          shape_dnf=down_sampled_shape,
                          send_pipe=send_pipe)

# ==========================================================================
# Instantiate C-Processes Running on LMT
# ==========================================================================
c_injector = PyToNxAdapter(shape=down_sampled_flat_shape)
c_spike_reader_multi_peak = NxToPyAdapter(shape=down_sampled_shape)
c_spike_reader_selective = NxToPyAdapter(shape=down_sampled_shape)

# ==========================================================================
# Instantiate Processes Running on Loihi 2
# ==========================================================================
sparse_1 = Sparse(weights=np.eye(num_neurons) * sparse1_weights)
dnf_multi_peak = LIF(shape=down_sampled_shape,
                     du=du_multipeak,
                     dv=dv_multipeak,
                     vth=multipeak_threshold)
connections_multi_peak = Sparse(weights=weights_multi_peak)
sparse_2 = Sparse(weights=np.eye(num_neurons) * sparse2_weights)
dnf_selective = LIF(shape=down_sampled_shape,
                    du=du_selective,
                    dv=dv_selective,
                    vth=selective_threshold)
connections_selective = Sparse(weights=weights_selective)

# ==========================================================================
# Connecting Processes
# ==========================================================================
# Connecting Input Processes
dvs_file_input.event_frame_out.connect(c_injector.in_port)
c_injector.out_port.connect(sparse_1.s_in)
sparse_1.a_out.reshape(new_shape=down_sampled_shape).connect(
    dnf_multi_peak.a_in)
dnf_multi_peak.s_out.reshape(new_shape=down_sampled_flat_shape).connect(
    sparse_2.s_in)
sparse_2.a_out.reshape(new_shape=down_sampled_shape).connect(
    dnf_selective.a_in)

# Recurrent-connecting MultiPeak DNF
con_ip = connections_multi_peak.s_in
dnf_multi_peak.s_out.reshape(new_shape=con_ip.shape).connect(con_ip)
con_op = connections_multi_peak.a_out
con_op.reshape(new_shape=dnf_multi_peak.a_in.shape).connect(
    dnf_multi_peak.a_in)

# Recurrent-connecting Selective DNF
con_ip = connections_selective.s_in
dnf_selective.s_out.reshape(new_shape=con_ip.shape).connect(con_ip)
con_op = connections_selective.a_out
con_op.reshape(new_shape=dnf_selective.a_in.shape).connect(
    dnf_selective.a_in)

# Connect C Reader Processes
dnf_multi_peak.s_out.connect(c_spike_reader_multi_peak.inp)
dnf_selective.s_out.connect(c_spike_reader_selective.inp)

# Connecting RateReaders
c_spike_reader_multi_peak.out.connect(rate_reader_multi_peak.in_port)
c_spike_reader_selective.out.connect(rate_reader_selective.in_port)

# Connecting ProcessOut (data relayer)
dvs_file_input.event_frame_out.reshape(
    new_shape=down_sampled_shape).connect(data_relayer.dvs_frame_port)
rate_reader_multi_peak.out_port.connect(data_relayer.dnf_multipeak_rates_port)
rate_reader_selective.out_port.connect(data_relayer.dnf_selective_rates_port)

# ==========================================================================
# Runtime Creation and Compilation
# ==========================================================================
exception_pm_map = {
    DVSFileInput: PyDVSFileInputPM,
    ProcessOut: DataRelayerPM
}
run_cfg = Loihi2HwCfg(exception_proc_model_map=exception_pm_map)
run_cnd = RunSteps(num_steps=num_steps, blocking=False)

# Compilation
compiler = Compiler()
executable = compiler.compile(dvs_file_input, run_cfg=run_cfg)

# Initializing runtime
mp = ActorType.MultiProcessing
runtime = Runtime(exe=executable,
                  message_infrastructure_type=mp)
runtime.initialize()


# ==========================================================================
# Bokeh Helpers
# ==========================================================================
def callback_run():
    runtime.start(run_condition=run_cnd)


def create_plot(plot_base_width, data_shape, title):
    x_range = DataRange1d(start=0,
                          end=data_shape[0],
                          bounds=(0, data_shape[0]),
                          range_padding=50,
                          range_padding_units='percent')
    y_range = DataRange1d(start=0,
                          end=data_shape[1],
                          bounds=(0, data_shape[1]),
                          range_padding=50,
                          range_padding_units='percent')

    pw = plot_base_width
    ph = int(pw * data_shape[1] / data_shape[0])
    plot = figure(plot_width=pw,
                  plot_height=ph,
                  x_range=x_range,
                  y_range=y_range,
                  match_aspect=True,
                  tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                  toolbar_location=None)

    image = plot.image([], x=0, y=0, dw=data_shape[0], dh=data_shape[1],
                       palette="Viridis256", level="image")

    plot.add_layout(Title(text=title, align="center"), "above")

    x_grid = list(range(data_shape[0]))
    plot.xgrid[0].ticker = x_grid
    y_grid = list(range(data_shape[1]))
    plot.ygrid[0].ticker = y_grid
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None

    color = LinearColorMapper(palette="Viridis256", low=0, high=1)
    image.glyph.color_mapper = color

    cb = ColorBar(color_mapper=color)
    plot.add_layout(cb, 'right')

    return plot, image

# ==========================================================================
# Instantiating Bokeh document
# ==========================================================================
bokeh_document = curdoc()

# create plots
dvs_frame_p, dvs_frame_im = create_plot(
    400, down_sampled_shape, "DVS file input (events)")
dnf_multipeak_rates_p, dnf_multipeak_rates_im = create_plot(
    400, down_sampled_shape, "DNF multi-peak (spike rates)")
dnf_selective_rates_p, dnf_selective_rates_im = create_plot(
    400, down_sampled_shape, "DNF selective (spike rates)")

# add a button widget and configure with the call back
button_run = Button(label="Run")
button_run.on_click(callback_run)

# finalize layout (with spacer as placeholder)
spacer = Spacer(height=40)
bokeh_document.add_root(
    gridplot([[button_run, None, None],
              [None, spacer, None],
              [dvs_frame_p, dnf_multipeak_rates_p, dnf_selective_rates_p]],
             toolbar_options=dict(logo=None)))


# ==========================================================================
# Bokeh Update
# ==========================================================================
def update(dvs_frame_ds_image,
           dnf_multipeak_rates_ds_image,
           dnf_selective_rates_ds_image):
    dvs_frame_im.data_source.data["image"] = [dvs_frame_ds_image]
    dnf_multipeak_rates_im.data_source.data["image"] = \
        [dnf_multipeak_rates_ds_image]
    dnf_selective_rates_im.data_source.data["image"] = \
        [dnf_selective_rates_ds_image]


# ==========================================================================
# Bokeh Main loop
# ==========================================================================
def main_loop():
    while True:
        data_for_plot_dict = recv_pipe.recv()
        bokeh_document.add_next_tick_callback(
            partial(update, **data_for_plot_dict)
        )


thread = Thread(target=main_loop)
thread.start()

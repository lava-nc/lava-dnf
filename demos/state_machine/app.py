# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from threading import Thread
from multiprocessing import Pipe
from functools import partial

from PIL import Image

from bokeh.plotting import curdoc
from bokeh.layouts import column, gridplot, layout, Spacer
from bokeh.models import ImageRGBA, Title, Button, Plot, ColumnDataSource, \
    Circle, Label, Arrow, NormalHead

from lava.proc.lif.process import LIF

from lava.proc.embedded_io.spike import PyToNxAdapter
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunContinuous

from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.kernels.kernels import SelectiveKernel
from lava.lib.dnf.operations.operations import Weights, ExpandDims, \
    ReduceDims, Convolution

from process_in.process import ProcessIn
from process_out.process import ProcessOut
from lava.lib.dnf.demos.state_machine.c_spike_reader.process import \
    CSpikeReader


# ==========================================================================
# Helpers
# ==========================================================================
def convert_digits_to_bias(
        combination: ty.Tuple[int, ...]
) -> ty.Tuple[np.ndarray, ...]:
    if len(combination) != 3:
        raise ValueError("'combination' must have exactly three entries, "
                         f"got: {combination=}.")

    biases = []
    for digit in combination:
        if not 0 <= digit <= 9:
            raise ValueError("Elements in 'combination' must be between"
                             f"0 and 9, got {digit}.")
        bias = np.zeros((10,), dtype=int)
        bias[digit] = 21
        biases.append(bias)
    return tuple(biases)


# ==========================================================================
# Params
# ==========================================================================
correct_combination = (1, 2, 3)

full_weight = 20
half_weight = 10
dnf_shape = (10,)
node_shape = (1,)
dnf_params = {"shape": dnf_shape, "vth": 19, "bias_exp": 6,
              "du": 4095, "dv": 4095}
node_params = {"shape": node_shape, "vth": 19, "bias_exp": 6,
               "du": 4095, "dv": 4095}

wm0_bias, wm1_bias, wm2_bias = convert_digits_to_bias(
    correct_combination)
cod_bias = [10] * 10

# inp_log_config = LogConfig(level=20, level_console=20)

input_kernel = SelectiveKernel(amp_exc=full_weight,
                               width_exc=0.1,
                               global_inh=0)

# ==========================================================================
# Instantiating Pipes
# ==========================================================================
# Creating Pipes to use for communication with Lava
recv_pipe_process_in, send_pipe_process_in = Pipe()
recv_pipe_process_out, send_pipe_process_out = Pipe()

# ==========================================================================
# Instantiating ProcessIn and ProcessOut
# ==========================================================================
process_in = ProcessIn(user_input_shape=dnf_shape,
                       recv_pipe=recv_pipe_process_in)
process_out = ProcessOut(dnf_shape=dnf_shape, node_shape=node_shape,
                         send_pipe=send_pipe_process_out)

# ==========================================================================
# Connecting Network
# ==========================================================================
injector = PyToNxAdapter(shape=dnf_shape)
c_spike_reader = CSpikeReader(dnf_shape=dnf_shape, node_shape=node_shape)

# ==========================================================================
# Instantiating Neuron Processes (LIF)
# ==========================================================================
task = LIF(**node_params, bias_mant=21)
task_int = LIF(**node_params)

beh0_int = LIF(**node_params)
beh0_cos = LIF(**node_params)

beh1_int = LIF(**node_params)
beh1_cos = LIF(**node_params)

beh2_int = LIF(**node_params)
beh2_cos = LIF(**node_params)

beh0_beh1_pc = LIF(**node_params)
beh1_beh2_pc = LIF(**node_params)

# Bias driven working-memory DNFs that store the correct combination of
# digits in sustained activation.
wm0 = LIF(**dnf_params, bias_mant=wm0_bias)
wm1 = LIF(**dnf_params, bias_mant=wm1_bias)
wm2 = LIF(**dnf_params, bias_mant=wm2_bias)

# Gate DNFs that can be boosted to direct activation of the working-memory
# DNFs into the CoS/CoD DNFs.
gate0 = LIF(**dnf_params)
gate1 = LIF(**dnf_params)
gate2 = LIF(**dnf_params)

# Checks for match between user input and the correct digits.
cos = LIF(**dnf_params)
# Checks for mismatch between user input and the correct digits.
cod = LIF(**dnf_params, bias_mant=cod_bias)
cod_node = LIF(**node_params)

# User input of digits.
inp = LIF(**dnf_params)

# ==========================================================================
# Connecting ProcessIn to CInjector to inp LIF population
# ==========================================================================
# ProcessIn to CInjector
process_in.user_input_out_port.connect(injector.inp)
# CInjector to inp LIF
connect(injector.out, inp.a_in, ops=[Weights(full_weight)])

# ==========================================================================
# Connecting Neurons to CSpikeReader to ProcessOut
# ==========================================================================
# Neurons to CSpikeReader
inp.s_out.connect(c_spike_reader.inp_in)

beh0_int.s_out.connect(c_spike_reader.beh0_int_in)
beh0_cos.s_out.connect(c_spike_reader.beh0_cos_in)

beh1_int.s_out.connect(c_spike_reader.beh1_int_in)
beh1_cos.s_out.connect(c_spike_reader.beh1_cos_in)

beh2_int.s_out.connect(c_spike_reader.beh2_int_in)
beh2_cos.s_out.connect(c_spike_reader.beh2_cos_in)

beh0_beh1_pc.s_out.connect(c_spike_reader.beh0_beh1_pc_in)
beh1_beh2_pc.s_out.connect(c_spike_reader.beh1_beh2_pc_in)

wm0.s_out.connect(c_spike_reader.wm0_in)
wm1.s_out.connect(c_spike_reader.wm1_in)
wm2.s_out.connect(c_spike_reader.wm2_in)

gate0.s_out.connect(c_spike_reader.gate0_in)
gate1.s_out.connect(c_spike_reader.gate1_in)
gate2.s_out.connect(c_spike_reader.gate2_in)

cos.s_out.connect(c_spike_reader.cos_in)
cod.s_out.connect(c_spike_reader.cod_in)

cod_node.s_out.connect(c_spike_reader.cod_node_in)

# CSpikeReader to ProcessOut
c_spike_reader.inp_out.connect(process_out.inp_in)

c_spike_reader.beh0_int_out.connect(process_out.beh0_int_in)
c_spike_reader.beh0_cos_out.connect(process_out.beh0_cos_in)

c_spike_reader.beh1_int_out.connect(process_out.beh1_int_in)
c_spike_reader.beh1_cos_out.connect(process_out.beh1_cos_in)

c_spike_reader.beh2_int_out.connect(process_out.beh2_int_in)
c_spike_reader.beh2_cos_out.connect(process_out.beh2_cos_in)

c_spike_reader.beh0_beh1_pc_out.connect(process_out.beh0_beh1_pc_in)
c_spike_reader.beh1_beh2_pc_out.connect(process_out.beh1_beh2_pc_in)

c_spike_reader.wm0_out.connect(process_out.wm0_in)
c_spike_reader.wm1_out.connect(process_out.wm1_in)
c_spike_reader.wm2_out.connect(process_out.wm2_in)

c_spike_reader.gate0_out.connect(process_out.gate0_in)
c_spike_reader.gate1_out.connect(process_out.gate1_in)
c_spike_reader.gate2_out.connect(process_out.gate2_in)

c_spike_reader.cos_out.connect(process_out.cos_in)
c_spike_reader.cod_out.connect(process_out.cod_in)

c_spike_reader.cod_node_out.connect(process_out.cod_node_in)

# ==========================================================================
# Connecting Network
# ==========================================================================
connect(inp.s_out, inp.a_in, ops=[Convolution(input_kernel)])

connect(wm0.s_out, gate0.a_in, ops=[Weights(half_weight)])
connect(wm1.s_out, gate1.a_in, ops=[Weights(half_weight)])
connect(wm2.s_out, gate2.a_in, ops=[Weights(half_weight)])

connect(gate0.s_out, cos.a_in, ops=[Weights(half_weight)])
connect(gate1.s_out, cos.a_in, ops=[Weights(half_weight)])
connect(gate2.s_out, cos.a_in, ops=[Weights(half_weight)])

connect(gate0.s_out, cod.a_in, ops=[Weights(-half_weight)])
connect(gate1.s_out, cod.a_in, ops=[Weights(-half_weight)])
connect(gate2.s_out, cod.a_in, ops=[Weights(-half_weight)])

connect(inp.s_out, cos.a_in, ops=[Weights(half_weight)])
connect(inp.s_out, cod.a_in, ops=[Weights(half_weight)])

connect(beh0_int.s_out, beh0_cos.a_in, ops=[Weights(half_weight)])
connect(beh0_cos.s_out, beh0_int.a_in, ops=[Weights(-full_weight)])
connect(beh0_cos.s_out, beh0_cos.a_in, ops=[Weights(full_weight)])

connect(beh1_int.s_out, beh1_cos.a_in, ops=[Weights(half_weight)])
connect(beh1_cos.s_out, beh1_int.a_in, ops=[Weights(-full_weight)])
connect(beh1_cos.s_out, beh1_cos.a_in, ops=[Weights(full_weight)])

connect(beh2_int.s_out, beh2_cos.a_in, ops=[Weights(half_weight)])
connect(beh2_cos.s_out, beh2_int.a_in, ops=[Weights(-full_weight)])
connect(beh2_cos.s_out, beh2_cos.a_in, ops=[Weights(full_weight)])

connect(beh0_cos.s_out, beh0_beh1_pc.a_in, ops=[Weights(-full_weight)])
connect(beh0_beh1_pc.s_out, beh1_int.a_in, ops=[Weights(-full_weight)])

connect(beh1_cos.s_out, beh1_beh2_pc.a_in, ops=[Weights(-full_weight)])
connect(beh1_beh2_pc.s_out, beh2_int.a_in, ops=[Weights(-full_weight)])

connect(beh0_int.s_out, gate0.a_in, ops=[Weights(half_weight),
                                         ExpandDims(dnf_shape)])
connect(beh1_int.s_out, gate1.a_in, ops=[Weights(half_weight),
                                         ExpandDims(dnf_shape)])
connect(beh2_int.s_out, gate2.a_in, ops=[Weights(half_weight),
                                         ExpandDims(dnf_shape)])

connect(cos.s_out, beh0_cos.a_in, ops=[Weights(half_weight),
                                       ReduceDims(0)])
connect(cos.s_out, beh1_cos.a_in, ops=[Weights(half_weight),
                                       ReduceDims(0)])
connect(cos.s_out, beh2_cos.a_in, ops=[Weights(half_weight),
                                       ReduceDims(0)])
connect(cos.s_out, inp.a_in, ops=[Weights(-full_weight)])

connect(cod.s_out, cod_node.a_in, ops=[Weights(full_weight),
                                       ReduceDims(0)])
connect(cos.s_out, cod_node.a_in, ops=[Weights(-full_weight),
                                       ReduceDims(0)])
connect(cod.s_out, inp.a_in, ops=[Weights(-full_weight)])
connect(cod_node.s_out, beh0_cos.a_in, ops=[Weights(-full_weight)])
connect(cod_node.s_out, beh1_cos.a_in, ops=[Weights(-full_weight)])
connect(cod_node.s_out, beh2_cos.a_in, ops=[Weights(-full_weight)])

connect(task.s_out, task_int.a_in, ops=[Weights(full_weight)])
connect(task_int.s_out, beh0_int.a_in, ops=[Weights(full_weight)])
connect(task_int.s_out, beh1_int.a_in, ops=[Weights(full_weight)])
connect(task_int.s_out, beh2_int.a_in, ops=[Weights(full_weight)])

connect(task.s_out, beh0_beh1_pc.a_in, ops=[Weights(full_weight)])
connect(task.s_out, beh1_beh2_pc.a_in, ops=[Weights(full_weight)])


# ==========================================================================
# Bokeh: Callback Helpers
# ==========================================================================
already_running = False


def callback_run():
    global already_running

    if already_running:
        print("Already running.")
    else:
        run_cfg = Loihi2HwCfg()
        run_cnd = RunContinuous()

        print("Running : BEGIN")
        inp.run(run_cfg=run_cfg, condition=run_cnd)
        print("Running : END")

        already_running = True


# def callback_stop():
#     global already_running
#
#     if already_running:
#         print("Stopping : BEGIN")
#         inp.stop()
#         print("Stopping : END")
#     else:
#         print("Nothing is running.")


def button_callback(button_number):
    # When a button is pressed, send the button number to Lava's ProcessIn
    # through the Pipe
    send_pipe_process_in.send(button_number)


# ==========================================================================
# Bokeh: Document
# ==========================================================================
bokeh_document = curdoc()

# ==========================================================================
# Bokeh: Run Button
# ==========================================================================
button_run = Button(label="Run", width=100, height=50)
button_run.on_click(callback_run)

# button_stop = Button(label="Stop", width=100, height=50)
# button_stop.on_click(callback_stop)

grid_control = gridplot([[button_run, None]],
                        toolbar_options=dict(logo=None))

# ==========================================================================
# Bokeh: Numpad Buttons
# ==========================================================================
button_0 = Button(label="0", width=70, height=70, button_type="primary")
button_0.on_click(partial(button_callback, 0))
button_1 = Button(label="1", width=70, height=70, button_type="primary")
button_1.on_click(partial(button_callback, 1))
button_2 = Button(label="2", width=70, height=70, button_type="primary")
button_2.on_click(partial(button_callback, 2))
button_3 = Button(label="3", width=70, height=70, button_type="primary")
button_3.on_click(partial(button_callback, 3))
button_4 = Button(label="4", width=70, height=70, button_type="primary")
button_4.on_click(partial(button_callback, 4))
button_5 = Button(label="5", width=70, height=70, button_type="primary")
button_5.on_click(partial(button_callback, 5))
button_6 = Button(label="6", width=70, height=70, button_type="primary")
button_6.on_click(partial(button_callback, 6))
button_7 = Button(label="7", width=70, height=70, button_type="primary")
button_7.on_click(partial(button_callback, 7))
button_8 = Button(label="8", width=70, height=70, button_type="primary")
button_8.on_click(partial(button_callback, 8))
button_9 = Button(label="9", width=70, height=70, button_type="primary")
button_9.on_click(partial(button_callback, 9))

grid_buttons = gridplot([[button_1, button_2, button_3],
                         [button_4, button_5, button_6],
                         [button_7, button_8, button_9],
                         [None, button_0, None]],
                        toolbar_options=dict(logo=None))

# ==========================================================================
# Bokeh: Circle and Plot Params for Neuron visualization
# ==========================================================================
# Shared Params
w_per_neuron = 25
lw = 3
h = 50
size = 15

# DNF Params
N = 10
x = np.linspace(0, 10, N)
w_dnf = w_per_neuron * (N) + w_per_neuron
fill_color_dnf = ["white" for i in range(N)]

# 1 Node
x_1 = [0]
w_1 = w_per_neuron * (2)
fill_color_1 = ["white"]

# 2 Node
x_2 = [0, 3]
w_2 = w_per_neuron * N // 2
fill_color_2 = ["white", "white"]

# ==========================================================================
# Bokeh: DataSources
# ==========================================================================
source_inp = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))

source_beh0 = ColumnDataSource(dict(x=x_2, fill_color=fill_color_2))
source_beh1 = ColumnDataSource(dict(x=x_2, fill_color=fill_color_2))
source_beh2 = ColumnDataSource(dict(x=x_2, fill_color=fill_color_2))

source_beh0_beh1_pc = ColumnDataSource(dict(x=x_1, fill_color=fill_color_1))
source_beh1_beh2_pc = ColumnDataSource(dict(x=x_1, fill_color=fill_color_1))

source_wm0 = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))
source_wm1 = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))
source_wm2 = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))

source_gate0 = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))
source_gate1 = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))
source_gate2 = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))

source_cos = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))
source_cod = ColumnDataSource(dict(x=x, fill_color=fill_color_dnf))

source_cod_node = ColumnDataSource(dict(x=x_1, fill_color=fill_color_1))

# ==========================================================================
# Bokeh: Plots
# ==========================================================================
plot_inp = Plot(title="Input", width=w_dnf, height=h, min_border=0,
                toolbar_location=None, match_aspect=True,
                outline_line_color=None)

plot_beh0 = Plot(title="Checking digit 1", width=w_dnf, height=h, min_border=0,
                 toolbar_location=None, match_aspect=True,
                 outline_line_color=None)
plot_beh1 = Plot(title="Checking digit 2", width=w_dnf, height=h, min_border=0,
                 toolbar_location=None, match_aspect=True,
                 outline_line_color=None)
plot_beh2 = Plot(title="Checking digit 3", width=w_dnf, height=h, min_border=0,
                 toolbar_location=None, match_aspect=True,
                 outline_line_color=None)

plot_beh0_beh1_pc = Plot(title=None, width=w_1, height=h, min_border=0,
                         toolbar_location=None, match_aspect=True,
                         outline_line_color=None)
plot_beh1_beh2_pc = Plot(title=None, width=w_1, height=h, min_border=0,
                         toolbar_location=None, match_aspect=True,
                         outline_line_color=None)

plot_wm0 = Plot(title="Correct digit 1", width=w_dnf, height=h, min_border=0,
                toolbar_location=None, match_aspect=True,
                outline_line_color=None)
plot_wm1 = Plot(title="Correct digit 2", width=w_dnf, height=h, min_border=0,
                toolbar_location=None, match_aspect=True,
                outline_line_color=None)
plot_wm2 = Plot(title="Correct digit 3", width=w_dnf, height=h, min_border=0,
                toolbar_location=None, match_aspect=True,
                outline_line_color=None)

plot_gate0 = Plot(title="Correct digit 1 (checking)", width=w_dnf, height=h,
                  min_border=0, toolbar_location=None, match_aspect=True,
                  outline_line_color=None)
plot_gate1 = Plot(title="Correct digit 2 (checking)", width=w_dnf, height=h,
                  min_border=0, toolbar_location=None, match_aspect=True,
                  outline_line_color=None)
plot_gate2 = Plot(title="Correct digit 3 (checking)", width=w_dnf, height=h,
                  min_border=0, toolbar_location=None, match_aspect=True,
                  outline_line_color=None)

plot_cos = Plot(title="Matching digit", width=w_dnf, height=h, min_border=0,
                toolbar_location=None, match_aspect=True,
                outline_line_color=None)
plot_cod = Plot(title="Non-matching digit", width=w_dnf, height=h, min_border=0,
                toolbar_location=None, match_aspect=True,
                outline_line_color=None)

plot_cod_node = Plot(title=None, width=w_dnf, height=h, min_border=0,
                     toolbar_location=None, match_aspect=True,
                     outline_line_color=None)

# ==========================================================================
# Bokeh: Labels and Titles
# ==========================================================================
citation_left = Label(x_offset=38, y_offset=2, x_units='screen',
                      y_units='screen', text_font_size="13px", text='Active')
plot_beh0.add_layout(citation_left, "left")
citation_right = Label(x_offset=-38, y_offset=2, x_units='screen',
                       y_units='screen', text_font_size="13px", text='Done')
plot_beh0.add_layout(citation_right, "right")

citation_left = Label(x_offset=38, y_offset=2, x_units='screen',
                      y_units='screen', text_font_size="13px", text='Active')
plot_beh1.add_layout(citation_left, "left")
citation_right = Label(x_offset=-38, y_offset=2, x_units='screen',
                       y_units='screen', text_font_size="13px", text='Done')
plot_beh1.add_layout(citation_right, "right")

citation_left = Label(x_offset=38, y_offset=2, x_units='screen',
                      y_units='screen', text_font_size="13px", text='Active')
plot_beh2.add_layout(citation_left, "left")
citation_right = Label(x_offset=-38, y_offset=2, x_units='screen',
                       y_units='screen', text_font_size="13px", text='Done')
plot_beh2.add_layout(citation_right, "right")

plot_beh0_beh1_pc.add_layout(Title(text="1 to 2", align="center"), "above")
plot_beh1_beh2_pc.add_layout(Title(text="2 to 3", align="center"), "above")

# ==========================================================================
# Bokeh: Circles
# ==========================================================================
glyph_inp = Circle(x="x", y=0, size=size, line_color="#3288bd",
                   fill_color="fill_color", line_width=lw)

glyph_beh0 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                    fill_color="fill_color", line_width=lw)
glyph_beh1 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                    fill_color="fill_color", line_width=lw)
glyph_beh2 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                    fill_color="fill_color", line_width=lw)

glyph_beh0_beh1_pc = Circle(x="x", y=0, size=size, line_color="#3288bd",
                            fill_color="fill_color", line_width=lw)
glyph_beh1_beh2_pc = Circle(x="x", y=0, size=size, line_color="#3288bd",
                            fill_color="fill_color", line_width=lw)

glyph_wm0 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                   fill_color="fill_color", line_width=lw)
glyph_wm1 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                   fill_color="fill_color", line_width=lw)
glyph_wm2 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                   fill_color="fill_color", line_width=lw)

glyph_gate0 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                     fill_color="fill_color", line_width=lw)
glyph_gate1 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                     fill_color="fill_color", line_width=lw)
glyph_gate2 = Circle(x="x", y=0, size=size, line_color="#3288bd",
                     fill_color="fill_color", line_width=lw)

glyph_cos = Circle(x="x", y=0, size=size, line_color="green",
                   fill_color="fill_color", line_width=lw)
glyph_cod = Circle(x="x", y=0, size=size, line_color="red",
                   fill_color="fill_color", line_width=lw)

glyph_cod_node = Circle(x="x", y=0, size=size, line_color="#3288bd",
                        fill_color="fill_color", line_width=lw)

# ==========================================================================
# Bokeh: Adding Circles and DataSources to Plots
# ==========================================================================
plot_inp.add_glyph(source_inp, glyph_inp)

plot_beh0.add_glyph(source_beh0, glyph_beh0)
plot_beh1.add_glyph(source_beh1, glyph_beh1)
plot_beh2.add_glyph(source_beh2, glyph_beh2)

plot_beh0_beh1_pc.add_glyph(source_beh0_beh1_pc, glyph_beh0_beh1_pc)
plot_beh1_beh2_pc.add_glyph(source_beh1_beh2_pc, glyph_beh1_beh2_pc)

plot_wm0.add_glyph(source_wm0, glyph_wm0)
plot_wm1.add_glyph(source_wm1, glyph_wm1)
plot_wm2.add_glyph(source_wm2, glyph_wm2)

plot_gate0.add_glyph(source_gate0, glyph_gate0)
plot_gate1.add_glyph(source_gate1, glyph_gate1)
plot_gate2.add_glyph(source_gate2, glyph_gate2)

plot_cos.add_glyph(source_cos, glyph_cos)
plot_cod.add_glyph(source_cod, glyph_cod)
plot_cod_node.add_glyph(source_cod_node, glyph_cod_node)

# ==========================================================================
# Bokeh: Numpad Image and Arrow
# ==========================================================================
# Load Image
numpad_pil_image = Image.open('numpad.png').convert('RGBA')
xdim, ydim = numpad_pil_image.size
img = np.empty((ydim, xdim), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
view[:, :, :] = np.flipud(np.asarray(numpad_pil_image))

# Numpad Image DataSource
numpad_image_data_source = ColumnDataSource(dict(image=[img]))
# Numpad Image Plot
plot_numpad_image = Plot(title=None, width=w_dnf, height=290, min_border=0,
                         toolbar_location=None, outline_line_color=None,
                         match_aspect=True)
# Numpad ImageRGBA
numpad_image = ImageRGBA(image="image", x=0, y=0, dw=w_dnf, dh=290)
# Adding ImageRGBA and DataSource to Plot
plot_numpad_image.add_glyph(numpad_image_data_source, numpad_image)

# ArrowHead
nh = NormalHead(fill_color="#3681c1", line_color="#3681c1", size=14)
# Arrow
arrow = Arrow(end=nh, line_color="#3681c1", line_width=5,
              x_start=w_dnf // 2, y_start=290, x_end=w_dnf // 2, y_end=340)
# Adding Arrow to Plot
plot_numpad_image.add_layout(arrow)

# ==========================================================================
# Bokeh: Neuron Layout
# ==========================================================================
# Spacers
spacer_neurons_1 = Spacer(height=40)
spacer_neurons_2 = Spacer(height=40)
spacer_neurons_3 = Spacer(height=20)
spacer_neurons_4 = Spacer(height=10)
spacer_neurons_5 = Spacer(height=10)

grid_plot = gridplot([
    [None, plot_beh0_beh1_pc, None, plot_beh1_beh2_pc, None],
    [plot_beh0, None, plot_beh1, None, plot_beh2],
    [None, None, spacer_neurons_1, None, None],
    [plot_wm0, None, plot_wm1, None, plot_wm2],
    [plot_gate0, None, plot_gate1, None, plot_gate2],
    [None, None, spacer_neurons_2, None, None],
    [None, None, plot_cos, None, None],
    [None, None, plot_cod, None, plot_cod_node],
    [None, None, spacer_neurons_3, None, None],
    [None, None, plot_inp, None, None],
    [None, None, plot_numpad_image, None, None],
], toolbar_options=dict(logo=None))

# ==========================================================================
# Bokeh: Complete Layout
# ==========================================================================
# Spacers
spacer_run_and_buttons = Spacer(height=100)
spacer_buttons_and_neurons = Spacer(width=100)
spacer_column_0 = Spacer(width=50)

column_1 = column(grid_control, spacer_run_and_buttons, grid_buttons)
column_2 = column(grid_plot)

bokeh_document.add_root(
    layout([[spacer_column_0, column_1, spacer_buttons_and_neurons, column_2]]))


# ==========================================================================
# Bokeh: Main loop and View Update Callback
# ==========================================================================
def update(inp_data,
           beh0_int_data, beh0_cos_data,
           beh1_int_data, beh1_cos_data,
           beh2_int_data, beh2_cos_data,
           beh0_beh1_pc_data, beh1_beh2_pc_data,
           wm0_data, wm1_data, wm2_data,
           gate0_data, gate1_data, gate2_data,
           cos_data,
           cod_data,
           cod_node_data):
    source_inp.data["fill_color"] = np.where(inp_data == 1., "#3288bd", "white")

    beh0_data = np.hstack((np.where(beh0_int_data == 1., "#3288bd", "white"),
                           np.where(beh0_cos_data == 1., "#3288bd", "white")))
    source_beh0.data["fill_color"] = beh0_data

    beh1_data = np.hstack((np.where(beh1_int_data == 1., "#3288bd", "white"),
                           np.where(beh1_cos_data == 1., "#3288bd", "white")))
    source_beh1.data["fill_color"] = beh1_data

    beh2_data = np.hstack((np.where(beh2_int_data == 1., "#3288bd", "white"),
                           np.where(beh2_cos_data == 1., "#3288bd", "white")))
    source_beh2.data["fill_color"] = beh2_data

    source_beh0_beh1_pc.data["fill_color"] = np.where(beh0_beh1_pc_data == 1.,
                                                      "#3288bd", "white")
    source_beh1_beh2_pc.data["fill_color"] = np.where(beh1_beh2_pc_data == 1.,
                                                      "#3288bd", "white")

    source_wm0.data["fill_color"] = np.where(wm0_data == 1.,
                                             "#3288bd", "white")
    source_wm1.data["fill_color"] = np.where(wm1_data == 1.,
                                             "#3288bd", "white")
    source_wm2.data["fill_color"] = np.where(wm2_data == 1.,
                                             "#3288bd", "white")

    source_gate0.data["fill_color"] = np.where(gate0_data == 1.,
                                               "#3288bd", "white")
    source_gate1.data["fill_color"] = np.where(gate1_data == 1.,
                                               "#3288bd", "white")
    source_gate2.data["fill_color"] = np.where(gate2_data == 1.,
                                               "#3288bd", "white")

    source_cos.data["fill_color"] = np.where(cos_data == 1.,
                                             "green", "white")
    source_cod.data["fill_color"] = np.where(cod_data == 1.,
                                             "red", "white")

    source_cod_node.data["fill_color"] = np.where(cod_node_data == 1.,
                                                  "#3288bd", "white")


def main_loop():
    while True:
        # Receive data from Lava's ProcessOut through the Pipe
        data_dict = recv_pipe_process_out.recv()
        # Use received data from Lava to update the Plots
        bokeh_document.add_next_tick_callback(partial(update, **data_dict))


thread = Thread(target=main_loop)
thread.start()
# ==========================================================================
# ==========================================================================

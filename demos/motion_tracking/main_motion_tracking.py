# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


from threading import Thread
from functools import partial
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, Spacer
from bokeh.models import LinearColorMapper, ColorBar, Title, Button
from bokeh.models.ranges import DataRange1d
import sys
from multiprocessing import Pipe
from motion_tracking_network import MotionNetwork

# ==========================================================================
# Parameters
# ==========================================================================
recv_pipe, send_pipe = Pipe()
num_steps = 10

# Checks whether terminate button has been clicked and allows to stop
# updating the bokeh doc
is_done = [False]
# ==========================================================================
# Set up network
# ==========================================================================
network = MotionNetwork(send_pipe,
                        num_steps)
# ==========================================================================
# Bokeh Helpers
# ==========================================================================


def callback_run():
    network.start()


def callback_stop():
    is_done[0] = True
    network.stop()
    sys.exit()


def create_plot(plot_base_width, data_shape, title, max_value=1):
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
    plot = figure(width=pw,
                  height=ph,
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

    color = LinearColorMapper(palette="Viridis256", low=0, high=max_value)
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
    400, network.downsampled_shape, "DVS file input (max pooling)",
    max_value=10)
dnf_multipeak_rates_p, dnf_multipeak_rates_im = create_plot(
    400, network.downsampled_shape, "DNF multi-peak (spike rates)")
dnf_selective_rates_p, dnf_selective_rates_im = create_plot(
    400, network.downsampled_shape, "DNF selective (spike rates)")

# add a button widget and configure with the call back
button_run = Button(label="Run")
button_run.on_click(callback_run)

button_stop = Button(label="Close")
button_stop.on_click(callback_stop)
# finalize layout (with spacer as placeholder)
spacer = Spacer(height=40)
bokeh_document.add_root(
    gridplot([[button_run, None, button_stop],
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
    while not is_done[0]:
        if recv_pipe.poll():
            data_for_plot_dict = recv_pipe.recv()
            bokeh_document.add_next_tick_callback(
                partial(update, **data_for_plot_dict))


thread = Thread(target=main_loop)
thread.start()
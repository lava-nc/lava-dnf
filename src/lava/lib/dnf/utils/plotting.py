# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import scipy.ndimage
import numpy as np
import typing as ty
import matplotlib.pyplot as plt
import matplotlib as mpl


# TODO (MR): This needs to be redesigned and properly unit tested.
def raster_plot(spike_data: np.ndarray,
                color: ty.Optional[str] = "rate",
                rate_window: ty.Optional[int] = 11) -> None:
    """
    Creates a raster plot, showing the spikes of all neurons over time.

    If the color parameter is set to "rate", the plot will use color to express
    the spike rate within a time window determined by rate_window (specified in
    number of time steps).

    Parameters
    ----------
    spike_data : numpy.ndarray
        spike data of dtype=bool (spike: 1, no-spike: 0) and shape (
        num_neurons, num_time_steps)
    color : str, optional
        color used for plotting spikes; if color is set to "rate" a
        color map is used to visualize the spike rate for each neuron in a time
        window
    rate_window : int, optional
        when setting color="rate", size of the time window in number of time
        steps

    """
    num_time_steps = np.size(spike_data, axis=0)
    # Generate a representation of spike times
    spike_times = np.arange(1, num_time_steps + 1)

    spike_rates = None

    if color == "rate":
        # Compute spike rates for each time window
        kernel = 1.0 / rate_window * np.ones((rate_window, 1))
        spike_rates = scipy.ndimage.convolve(spike_data, kernel)
        c = []
    else:
        # Set the given color as the color for the plot
        c = color

    # Lists to hold the x and y values of the plot
    x = []
    y = []

    for time_idx, time_step in enumerate(spike_times):
        # Determine all neurons that spiked in this time step
        spiking_neuron_idx, = np.where(spike_data[time_idx, :] == 1.0)
        if color == "rate":
            # Add the spike rate values of the spiking neurons at the current
            # time step to c
            c.extend(spike_rates[time_idx, spiking_neuron_idx])
        spiking_neuron_idx = spiking_neuron_idx.tolist()
        # If any neurons spiked...
        if len(spiking_neuron_idx) > 0:
            # ...add the index of all spiking neurons to y
            y.extend(spiking_neuron_idx)
            # ...add the current time step to x (as many times as there are
            # spiking neurons)
            x.extend(len(spiking_neuron_idx) * [time_step])

    # Generate the plot
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.antialiased'] = False
    plt.scatter(x, y, c=c, marker='|', s=5)

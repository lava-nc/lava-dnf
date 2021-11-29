# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import scipy.ndimage
import numpy as np
import typing as ty
import matplotlib.pyplot as plt
import matplotlib as mpl


def compute_spike_rates(spike_data: np.ndarray,
                        window_size: ty.Optional[int] = 11) -> np.ndarray:
    """
    Computes instantaneous spike rates for all neurons over time

    This method uses the window_size parameter to derive a kernel with which
    it convolves the spike_data.
    Yields an array of the same shape, with each value representing the spike
    rate of a neuron over the specified time window.

    Parameters
    ----------
    spike_data : numpy.ndarray
        spike data of dtype=bool (spike: 1, no-spike: 0) and
        shape (num_time_steps, num_neurons)
    window_size : int, optional
        size of the time window in number of time steps

    Returns
    -------
    spike_rates : numpy.ndarray
        array of same shape as spike_data which represents the instantaneous
        spike rate of every neuron at every time step
    """
    # Compute spike rates for each time window
    kernel = np.ones((window_size, 1)) / window_size
    spike_rates = scipy.ndimage.convolve(spike_data, kernel)

    return spike_rates


def _compute_colored_spike_coordinates(spike_data: np.ndarray,
                                       spike_rates: np.ndarray) -> \
        ty.Tuple[ty.List[int], ty.List[int], ty.List[float]]:
    """
    Computes coordinates of each spike to be shown in a raster plot, along
    with a color translating the instantaneous spike rate of the neuron
    at the time where it spiked.

    Parameters
    ----------
    spike_data : numpy.ndarray
        spike data of dtype=bool (spike: 1, no-spike: 0) and
        shape (num_time_steps, num_neurons)
    spike_rates : numpy.ndarray
        array of same shape as spike_data which represents the instantaneous
        spike rate of every neuron at every time step

    Returns
    -------
    x : list(int)
        list of x coordinates of all spikes in the to-be shown plot
    y : list(int)
        list of y coordinates of all spikes in the to-be shown plot
    colors : list(float)
        list of colors (based on instantaneous spike rates) of all spikes in
        the to-be shown plot
    """
    num_time_steps = spike_data.shape[0]
    # Generate a representation of spike times
    time_array = np.arange(1, num_time_steps + 1)

    # Lists to hold the x and y values of the plot
    x = []
    y = []
    colors = []

    for time_idx, time_step in enumerate(time_array):
        # Determine all neurons that spiked in this time step
        spiking_neuron_idx, = np.where(spike_data[time_idx, :] == 1.0)

        # Add the spike rate values of the spiking neurons at the current
        # time step to c
        colors.extend(spike_rates[time_idx, spiking_neuron_idx])

        spiking_neuron_idx = spiking_neuron_idx.tolist()
        # If any neurons spiked...
        if len(spiking_neuron_idx) > 0:
            # ...add the index of all spiking neurons to y
            y.extend(spiking_neuron_idx)
            # ...add the current time step to x (as many times as there are
            # spiking neurons)
            x.extend(len(spiking_neuron_idx) * [time_step])

    return x, y, colors


def raster_plot(spike_data: np.ndarray,
                window_size: ty.Optional[int] = 11) -> None:
    """
    Creates a raster plot, showing the spikes of all neurons over time.

    The plot will use color to express the spike rate within a time window
    determined by rate_window (specified in number of time steps).

    Parameters
    ----------
    spike_data : numpy.ndarray
        spike data of dtype=bool (spike: 1, no-spike: 0) and
        shape (num_time_steps, num_neurons)
    window_size : int, optional
        size of the time window in number of time steps
    """
    spike_rates = compute_spike_rates(spike_data, window_size)

    x, y, colors = _compute_colored_spike_coordinates(spike_data, spike_rates)

    # Generate the plot
    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.antialiased'] = False

    plt.scatter(x, y, c=colors, marker='|', s=5)

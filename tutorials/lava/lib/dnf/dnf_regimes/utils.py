# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display
import typing as ty

from lava.lib.dnf.utils.plotting import raster_plot, compute_spike_rates


def plot_1d(probe_data_dnf: np.ndarray,
            probe_data_input1: np.ndarray,
            probe_data_input2: np.ndarray) -> None:
    """Generates an architecture raster plot for the examples in the DNF 101
    tutorial.

    Parameters
    ----------
    probe_data_dnf : numpy.ndarray
        probe data of the DNF
    probe_data_input1 : numpy.ndarray
        probe data of the first spiking input
    probe_data_input2 : numpy.ndarray
        probe data of the second spiking input
    """
    probe_data_input = probe_data_input1 + probe_data_input2
    probe_data_input = probe_data_input.astype(float)
    probe_data_input = np.transpose(probe_data_input)
    probe_data_dnf = np.transpose(probe_data_dnf.astype(float))

    num_neurons = np.size(probe_data_input, axis=1)
    num_time_steps = np.size(probe_data_input, axis=0)

    plt.figure(figsize=(10, 5))
    ax0 = plt.subplot(2, 1, 1)
    raster_plot(probe_data_input)
    ax0.set_xlabel(None)
    ax0.set_ylabel('Input\nNeuron idx')
    ax0.set_xticklabels([])
    ax0.set_yticks([0, num_neurons - 1])
    ax0.set_xlim(0, num_time_steps)
    ax0.set_ylim(-1, num_neurons)

    ax1 = plt.subplot(2, 1, 2)
    raster_plot(probe_data_dnf)
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('DNF\nNeuron idx')
    ax1.set_yticks([0, num_neurons - 1])
    ax1.set_xlim(0, num_time_steps)
    ax1.set_ylim(-1, num_neurons)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.035, 0.8])
    plt.colorbar(cax=cax, label="Spike rate")

    plt.show()


def animated_1d_plot(probe_data_dnf: np.ndarray,
                     probe_data_input1: np.ndarray,
                     probe_data_input2: np.ndarray,
                     interval: ty.Optional[int] = 30) -> None:
    """Generates an animated plot for examples in the DNF regimes tutorial.

    Parameters
    ----------
    probe_data_dnf : numpy.ndarray
        probe data of the DNF
    probe_data_input1 : numpy.ndarray
        probe data of the first spiking input
    probe_data_input2 : numpy.ndarray
        probe data of the second spiking input
    interval : int
        interval to use in matplotlib.animation.FuncAnimation
    """
    probe_data_input = probe_data_input1 + probe_data_input2
    probe_data_input = probe_data_input.astype(float)
    probe_data_dnf = probe_data_dnf.astype(float)
    probe_data_input = np.transpose(probe_data_input)
    probe_data_dnf = np.transpose(probe_data_dnf)

    num_neurons = np.size(probe_data_input, axis=1)
    num_time_steps = np.size(probe_data_dnf, axis=0)

    input_spike_rates = compute_spike_rates(probe_data_input)
    dnf_spike_rates = compute_spike_rates(probe_data_dnf)

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    line0, = ax[0].plot(np.zeros((num_neurons,)), 'bo-')
    line1, = ax[1].plot(np.zeros((num_neurons,)), 'ro-')

    im = [line0, line1]

    ax[0].set_xlabel("")
    ax[1].set_xlabel("Input neuron idx")

    ax[0].set_ylabel("Input spike rate")
    ax[1].set_ylabel("DNF spike rate")

    ax[0].set_xticks([])
    ax[1].set_xticks([0, num_neurons - 1])

    ax[0].set_yticks([0, 1])
    ax[1].set_yticks([0, 1])

    ax[0].set_xlim(-1, num_neurons)
    ax[1].set_xlim(-1, num_neurons)

    offset = 0.1
    ax[0].set_ylim(np.min(input_spike_rates) - offset,
                   np.max(input_spike_rates) + offset)
    ax[1].set_ylim(np.min(dnf_spike_rates) - offset,
                   np.max(dnf_spike_rates) + offset)

    plt.tight_layout()

    def animate(i: int) -> ty.List:
        x = range(num_neurons)
        im[0].set_data(x, input_spike_rates[i, :])
        im[1].set_data(x, dnf_spike_rates[i, :])
        return im

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   frames=num_time_steps,
                                   interval=interval,
                                   blit=True)

    html = display.HTML(anim.to_jshtml())

    display.display(html)
    plt.close()

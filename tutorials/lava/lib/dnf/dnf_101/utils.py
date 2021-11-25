# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import matplotlib.pyplot as plt

from lava.lib.dnf.utils.plotting import raster_plot


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
    probe_data_input = probe_data_input.astype(np.float)
    probe_data_input = np.transpose(probe_data_input)
    probe_data_dnf = np.transpose(probe_data_dnf.astype(np.float))

    num_neurons = np.size(probe_data_input, axis=1)
    num_time_steps = np.size(probe_data_input, axis=0)

    plt.figure(figsize=(7, 3.5))
    ax0 = plt.subplot(2, 1, 1)
    raster_plot(probe_data_input)
    ax0.set_xlabel(None)
    ax0.set_ylabel('Input\nNeuron idx')
    ax0.set_xticklabels([])
    ax0.set_yticks([0, num_neurons-1])
    ax0.set_xlim(0, num_time_steps)
    ax0.set_ylim(-1, num_neurons)

    ax1 = plt.subplot(2, 1, 2)
    raster_plot(probe_data_dnf)
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('DNF\nNeuron idx')
    ax1.set_yticks([0, num_neurons-1])
    ax1.set_xlim(0, num_time_steps)
    ax1.set_ylim(-1, num_neurons)

    plt.tight_layout()
    plt.show()

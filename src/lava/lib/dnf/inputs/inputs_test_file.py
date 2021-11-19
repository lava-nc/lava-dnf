# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava.lib.dnf.inputs.patterns.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.generators.spike_generator.process import SpikeGenerator


def main():
    num_steps = 100
    shape = (60,)

    gauss_pattern = GaussPattern(shape=shape, amplitude=1500.0, mean=40, stddev=15)
    spike_generator = SpikeGenerator(shape=shape)

    gauss_pattern.out_ports.a_out.connect(spike_generator.in_ports.a_in)

    spikes = np.zeros((shape[0], 3 * num_steps))

    for i in range(num_steps):
        print(f"Step {i}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                            run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i] = spike_generator.vars.spikes.get()

    gauss_pattern.mean = 30

    for i in range(num_steps):
        print(f"Step {i + num_steps}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                            run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i + num_steps] = spike_generator.vars.spikes.get()

    gauss_pattern.mean = 20

    for i in range(num_steps):
        print(f"Step {i + (2 * num_steps)}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                            run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i + (2 * num_steps)] = spike_generator.vars.spikes.get()

    spike_generator.stop()

    plt.figure(figsize=(20, 20), facecolor="grey")

    for idx, spike_array in enumerate(spikes):
        spike_positions = np.where(spike_array == 1.0)
        plt.eventplot(spike_positions, lineoffsets=idx, linelengths=0.5)

    plt.show()


if __name__ == "__main__":
    main()

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.dnf.inputs.spike_source.process import SpikeSource
from lava.lib.dnf.inputs.spike_source.models import SpikeSourceProcessModel
from lava.lib.dnf.inputs.inputs import GaussInputPattern, SpikeInputGenerator
from lava.magma.core.model.py.type import LavaPyType


class TestPopulationSubProcessModel(unittest.TestCase):
    def test_init(self):
        """Tests creation of a SpikeSourceProcessModel."""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)
        spike_source = SpikeSource(generator=spike_generator)

        pm = SpikeSourceProcessModel(spike_source)

        self.assertIsInstance(pm, SpikeSourceProcessModel)

    def test_spike_generator_accessible(self):
        """Tests whether the spike generator is accessible as a property of the process model and is of
            the right type."""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)
        spike_source = SpikeSource(generator=spike_generator)

        pm = SpikeSourceProcessModel(spike_source)

        self.assertIsInstance(pm.spike_generator, SpikeInputGenerator)

    def test_ports_exist(self):
        """Tests whether the LavaPyType port is created."""
        input_pattern = GaussInputPattern(shape=(60,), amplitude=2.0, mean=30, stddev=3)
        spike_generator = SpikeInputGenerator(input_pattern)
        spike_source = SpikeSource(generator=spike_generator)

        pm = SpikeSourceProcessModel(spike_source)

        self.assertIsInstance(pm.s_out, LavaPyType)


if __name__ == '__main__':
    unittest.main()

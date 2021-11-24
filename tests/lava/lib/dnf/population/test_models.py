# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.lif.process import LIF
from lava.lib.dnf.population.process import Population
from lava.lib.dnf.population.models import PopulationSubProcessModel


class TestPopulationSubProcessModel(unittest.TestCase):
    def test_init_with_default_arguments(self) -> None:
        """Tests creation of a PopulationSubProcessModel."""
        pm = PopulationSubProcessModel(Population())
        self.assertIsInstance(pm, PopulationSubProcessModel)

    def test_lif_process_init(self) -> None:
        """Tests creation of sub process (LIF)."""
        population = Population(shape=(5, 3))
        pm = PopulationSubProcessModel(population)
        self.assertIsInstance(pm.neurons, LIF)

    def test_ports_are_connected_to_process_ports(self) -> None:
        """Tests whether the InPorts and OutPorts of the Population process
        are correctly connected to the InPorts and OutPorts of the
        LIF neurons within the ProcessModel."""
        population = Population()
        pm = PopulationSubProcessModel(population)

        # check whether the InPort of Population is connected to the
        # Inport of LIF
        pop_ip = population.in_ports.a_in
        lif_ip = pm.neurons.in_ports.a_in
        self.assertEqual(pop_ip.get_dst_ports(), [lif_ip])
        self.assertEqual(lif_ip.get_src_ports(), [pop_ip])

        # check whether the OutPort of LIF is connected
        # to the OutPort of Population
        lif_op = pm.neurons.out_ports.s_out
        pop_op = population.out_ports.s_out
        self.assertEqual(lif_op.get_dst_ports(), [pop_op])
        self.assertEqual(pop_op.get_src_ports(), [lif_op])


if __name__ == '__main__':
    unittest.main()

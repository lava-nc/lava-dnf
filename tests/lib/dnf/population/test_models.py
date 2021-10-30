# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.proc.lif.process import LIF
from lib.dnf.population.process import Population
from lib.dnf.population.models import PopulationSubProcessModel


class TestPopulationSubProcessModel(unittest.TestCase):
    def test_init_with_default_arguments(self):
        """Tests creation of a PopulationSubProcessModel."""
        pm = PopulationSubProcessModel(Population())
        self.assertIsInstance(pm, PopulationSubProcessModel)

    def test_lif_process_init(self):
        """Tests creation of sub process (LIF)."""
        population = Population(shape=(5, 3))
        pm = PopulationSubProcessModel(population)
        self.assertIsInstance(pm.neurons, LIF)

    def test_ports_are_connected_to_process_ports(self):
        """Tests whether the InPorts and OutPorts of the Population process
        are correctly connected to the InPorts and OutPorts of the
        LIF neurons within the ProcessModel."""
        population = Population()
        pm = PopulationSubProcessModel(population)

        # check whether the InPort of Population is connected to the
        # Inport of LIF
        pop_ip = population.in_ports.a_in
        lif_ip = pm.neurons.in_ports.a_in
        self.assertEqual(pop_ip.out_connections, [lif_ip])
        self.assertEqual(lif_ip.in_connections, [pop_ip])

        # check whether the OutPort of LIF is connected
        # to the OutPort of Population
        lif_op = pm.neurons.out_ports.s_out
        pop_op = population.out_ports.s_out
        self.assertEqual(lif_op.out_connections, [pop_op])
        self.assertEqual(pop_op.in_connections, [lif_op])


class TestTauToDecay(unittest.TestCase):
    def setUp(self) -> None:
        """Creates a PopulationSubProcessModel instance to have access
        to the _tau_to_decay method."""
        self.pm = PopulationSubProcessModel(Population())

    def test_positive_tau_values_are_converted_to_decays(self):
        """Tests whether positive tau values are converted to decays."""
        self.assertEqual(self.pm._tau_to_decay(1), 4095)
        self.assertEqual(self.pm._tau_to_decay(4095), 1)

    def test_tau_values_below_one_raise_value_error(self):
        """Tests whether setting tau<1 raises a ValueError."""
        with self.assertRaises(ValueError):
            self.pm._tau_to_decay(-10)
        with self.assertRaises(ValueError):
            self.pm._tau_to_decay(0)
        with self.assertRaises(ValueError):
            self.pm._tau_to_decay(0.9)

    def test_float_tau_values_are_cast_to_int(self):
        """Tests whether decays are cast to int even if tau is type float."""
        self.assertIsInstance(self.pm._tau_to_decay(10.5), int)


if __name__ == '__main__':
    unittest.main()

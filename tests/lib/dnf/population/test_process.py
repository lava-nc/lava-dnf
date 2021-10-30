# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.process.ports.ports import InPort, OutPort
from lib.dnf.population.process import Population


class TestPopulations(unittest.TestCase):
    def test_init(self):
        """Tests whether a population process can be initiated."""
        population = Population()
        self.assertIsInstance(population, Population)

    def test_name(self):
        """Tests the name argument."""
        test_name = "Test name"
        population = Population(name=test_name)
        self.assertEqual(population.name, test_name)

    def test_default_shape(self):
        """Tests whether shape is set by default."""
        population = Population()
        self.assertEqual(population.shape, (1,))

    def test_setting_shape(self):
        """Tests whether setting the shape works."""
        shape = (5, 3)
        population = Population(shape=shape)
        self.assertEqual(population.shape, shape)

    def test_ports_exist(self):
        """Tests whether the InPort and OutPort is created."""
        shape = (5, 3)
        population = Population(shape=shape)
        self.assertIsInstance(population.in_ports.a_in, InPort)
        self.assertIsInstance(population.out_ports.s_out, OutPort)
        self.assertEqual(population.in_ports.a_in.shape, shape)
        self.assertEqual(population.out_ports.s_out.shape, shape)


if __name__ == '__main__':
    unittest.main()

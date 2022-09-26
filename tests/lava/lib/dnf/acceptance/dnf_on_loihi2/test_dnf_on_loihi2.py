# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.lib.dnf.inputs.gauss_pattern.models import GaussPatternProcessModel
from lava.lib.dnf.inputs.rate_code_spike_gen.models import \
    RateCodeSpikeGenProcessModel

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from lava.lib.dnf.kernels.kernels import SelectiveKernel, MultiPeakKernel
from lava.lib.dnf.operations.operations import Convolution, Weights
from lava.lib.dnf.connect.connect import connect
from lava.proc.dense.models import PyDenseModelBitAcc
from lava.proc.lif.models import PyLifModelBitAcc
from lava.proc.lif.ncmodels import NcModelLifHC
from lava.proc.dense.ncmodels import NcModelDense
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.c.type import LavaCType, LavaCDataType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import LMT
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.c.model import CLoihiProcessModel
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.process.ports.ports import (
    InPort, OutPort
)
from lava.magma.core.model.c.ports import CInPort, COutPort
from lava.magma.core.sync.domain import SyncDomain


class Injector(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.in_port = InPort(shape)
        self.out_port = OutPort(shape)


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(LMT)
class InjectorModel(CLoihiProcessModel):
    in_port: COutPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    out_port: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)

    @property
    def source_file_name(self) -> str:
        return "injector.c"


class TestDNFOnLoihi2(unittest.TestCase):
    def test_1d_dnf_on_loihi2(self) -> None:
        num_steps = 10
        num_neurons = 10

        shape = (num_neurons,)

        dnf_params = {"shape": shape,
                      "du": 409,
                      "dv": 2047,
                      "vth": 200}
        # kernel_params = {"amp_exc": 18,
        #                  "width_exc": 3,
        #                  "global_inh": -15}
        kernel_params = {"amp_exc": 83,
                         "width_exc": 3.75,
                         "amp_inh": -70,
                         "width_inh": 7.5}

        bias_mant = np.zeros((num_neurons,), dtype=np.int32)
        bias_mant[3:6] = 100
        input_lif_params = {"shape": shape,
                            "bias_mant": bias_mant,
                            "bias_exp": 6,
                            "du": 0,
                            "dv": 0,
                            "vth": 20}
        input_dense_params = {"weights": np.eye(shape[0], dtype=int) * 25}

        ### Python ###

        dnf = LIF(**dnf_params)
        #kernel = SelectiveKernel(**kernel_params)
        kernel = MultiPeakKernel(**kernel_params)
        dense = connect(dnf.s_out, dnf.a_in, ops=[Convolution(kernel)])

        # LIF providing input to the recurrent LIF via symmetric Dense.
        input_lif = LIF(**input_lif_params)
        input_dense = Dense(**input_dense_params)
        input_lif.s_out.connect(input_dense.s_in)
        input_dense.a_out.connect(dnf.a_in)


        # # GaussPattern produces a pattern of spike rates
        # gauss_pattern = GaussPattern(shape=shape, amplitude=1000, mean=7,
        #                              stddev=5)
        #
        # # The spike generator produces spikes based on the spike rates given
        # # by the Gaussian pattern
        # spike_generator = RateCodeSpikeGen(shape=shape)
        # gauss_pattern.a_out.connect(spike_generator.a_in)
        #
        # # Connect the spike generator to a population
        # connect(spike_generator.s_out, dnf.a_in, ops=[Weights(20)])

        pmm = {LIF: PyLifModelBitAcc,
               Dense: PyDenseModelBitAcc}#,
               # GaussPattern: GaussPatternProcessModel,
               # RateCodeSpikeGen: RateCodeSpikeGenProcessModel}

        voltages_py = np.zeros((num_neurons, num_steps), dtype=int)
        try:
            # Start running the network (explained below)
            for i in range(num_steps):
                dnf.run(condition=RunSteps(num_steps=1),
                        run_cfg=Loihi1SimCfg(exception_proc_model_map=pmm))
                voltages_py[:, i] = dnf.v.get()
        finally:
            # Stop the run to free resources
            dnf.stop()

        print(f"{voltages_py}")

        ### LOIHI 2 ###

        dnf = LIF(**dnf_params)
        #kernel = SelectiveKernel(**kernel_params)
        kernel = MultiPeakKernel(**kernel_params)
        dense = connect(dnf.s_out, dnf.a_in, ops=[Convolution(kernel)])

        input_lif = LIF(**input_lif_params)
        input_dense = Dense(**input_dense_params)
        input_lif.s_out.connect(input_dense.s_in)
        input_dense.a_out.connect(dnf.a_in)

        # # GaussPattern produces a pattern of spike rates
        # gauss_pattern = GaussPattern(shape=shape, amplitude=1000, mean=7,
        #                              stddev=5)
        #
        # # The spike generator produces spikes based on the spike rates given
        # # by the Gaussian pattern
        # spike_generator = RateCodeSpikeGen(shape=shape)
        # gauss_pattern.a_out.connect(spike_generator.a_in)
        #
        # # Connect the spike generator to a population
        # injector = Injector(shape=shape)
        # spike_generator.s_out.connect(injector.in_port)
        # connect(injector.out_port, dnf.a_in, ops=[Weights(20)])

        #injector.out_port.connect(dense2.s_in)
        #dense2.a_out.connect(dnf.a_in)

        pmm = {LIF: NcModelLifHC,
               Dense: NcModelDense}#,
               # GaussPattern: GaussPatternProcessModel,
               # RateCodeSpikeGen: RateCodeSpikeGenProcessModel}

        voltages_nc = np.zeros((num_neurons, num_steps), dtype=int)
        try:
            # Start running the network (explained below)
            for i in range(num_steps):
                print(f"{i=}")
                dnf.run(condition=RunSteps(num_steps=1),
                        run_cfg=Loihi2HwCfg(exception_proc_model_map=pmm))
                voltages_nc[:, i] = dnf.v.get()
        finally:
            # Stop the run to free resources
            dnf.stop()

        voltages_nc = np.where(voltages_nc == 128, 0, voltages_nc)

        print(f"{voltages_py}")
        print(f"{voltages_nc}")

        np.testing.assert_array_equal(voltages_py, voltages_nc)


if __name__ == '__main__':
    unittest.main()

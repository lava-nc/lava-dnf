# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod
import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess


class SpikeSource(AbstractProcess):
    def __init__(self,
                 spike_gen: SpikeGenerator):
        self._spike_gen = spike_gen
        self.s_out = OutPort()

    # would go into the Process Model
    def run_spk(self):
        self.proc._spike_gen.generate()


class BiasSource(AbstractProcess):
    def __init__(self,
                 bias_gen: BiasGenerator):
        self._bias_gen = bias_gen
        self._bias_ref = RefPort()


class InputGenerator(ABC):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 input_param: InputParam):
        self._shape = shape
        self._input_param = input_param

    @abstractmethod
    def generate(self, *args) -> np.ndarray:
        pass


class SpikeGenerator(InputGenerator):
    def generate(self, time_step) -> np.ndarray:
        spike_rates = self._input_param(self._shape)
        return self._to_spikes(spike_rates, time_step)

    def _to_spikes(self, spike_rates: np.ndarray, time_step) -> np.ndarray:
        pass  # TODO


class BiasGenerator(InputGenerator):
    def generate(self) -> np.ndarray:
        return self._input_param(self._shape)


class InputParam(ABC):
    def __init__(self, function, **kwargs):
        self._function = function
        self._params = kwargs

    def __call__(self, shape) -> np.ndarray:
        return self._function(shape, self._params)


class GaussParam(InputParam):
    def __init__(self, **kwargs):
        self._kwargs = self._validate_kwargs(kwargs)
        super().__init__(gauss, self._kwargs)

    def _validate_kwargs(self, kwargs):
        pass  # TODO



params = GaussParam(shape=(20, 20), max_rate=0, center=[11, 11], width=[2, 2])
spike_gen = SpikeGenerator(params)
spike_src = SpikeSourceProcess(spike_gen)
network.run(50)
pattern.max_rate = 2000
network.run(100)

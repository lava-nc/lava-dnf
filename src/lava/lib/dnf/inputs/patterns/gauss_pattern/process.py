# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort


class GaussPattern(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        shape = kwargs.pop("shape")
        amplitude = kwargs.pop("amplitude")
        mean = self._validate_mean(np.array(shape), kwargs.pop("mean"))
        stddev = self._validate_stddev(np.array(shape), kwargs.pop("stddev"))

        self._shape = Var(shape=(len(shape),), init=np.array(shape))

        self._amplitude = Var(shape=(1,), init=np.array([amplitude]))
        self._mean = Var(shape=(len(shape),), init=mean)
        self._stddev = Var(shape=(len(shape),), init=stddev)

        self.pattern = Var(shape=shape, init=np.zeros(shape))

        self.changed = Var(shape=(1,), init=np.array([True]))

        self.a_out = OutPort(shape=shape)

    def _validate_mean(self,
                       shape: np.ndarray,
                       mean: ty.Union[float, ty.List[float]]) -> np.ndarray:
        if isinstance(mean, float) or isinstance(mean, int):
            mean = float(mean)
            mean = [mean]

        if len(mean) == 1:
            mean_val = mean[0]
            mean = [mean_val for _ in range(shape.shape[0])]
        elif len(mean) > 1:
            if len(mean) != shape.shape[0]:
                raise ValueError("<mean> parameter has length different from shape dimensionality")
        else:
            raise ValueError("<mean> parameter cannot be empty")

        return np.array(mean)

    def _validate_stddev(self,
                         shape: np.ndarray,
                         stddev: ty.Union[float, ty.List[float]]) -> np.ndarray:
        if isinstance(stddev, float) or isinstance(stddev, int):
            stddev = float(stddev)
            stddev = [stddev]

        if len(stddev) == 1:
            stddev_val = stddev[0]
            stddev = [stddev_val for _ in range(shape.shape[0])]
        elif len(stddev) > 1:
            if len(stddev) != shape.shape[0]:
                raise ValueError("<stddev> parameter has length different from shape dimensionality")
        else:
            raise ValueError("<stddev> parameter cannot be empty")

        return np.array(stddev)

    def _update(self):
        self.changed.set(np.array([True]))

    @property
    def shape(self) -> np.ndarray:
        try:
            return self._shape.get()
        except AttributeError:
            return None

    @property
    def amplitude(self) -> np.ndarray:
        try:
            return self._amplitude.get()
        except AttributeError:
            return None

    @amplitude.setter
    def amplitude(self, amplitude: float):
        self._amplitude.set(np.array([amplitude]))
        self._update()

    @property
    def mean(self) -> np.ndarray:
        try:
            return self._mean.get()
        except AttributeError:
            return None

    @mean.setter
    def mean(self, mean: ty.Union[float, ty.List[float]]):
        mean = self._validate_mean(self.shape, mean)
        self._mean.set(mean)
        self._update()

    @property
    def stddev(self) -> np.ndarray:
        try:
            return self._stddev.get()
        except AttributeError:
            return None

    @stddev.setter
    def stddev(self, stddev: ty.Union[float, ty.List[float]]):
        stddev = self._validate_stddev(self.shape, stddev)
        self._stddev.set(stddev)
        self._update()

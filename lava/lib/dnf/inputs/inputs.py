# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod
import numpy as np
import typing as ty

from typing import Tuple, Optional, Union
import scipy.stats


def gauss(shape: Tuple[int],
          domain: Optional[np.ndarray] = None,
          amplitude: float = 1.0,
          mean: Optional[Union[float, np.ndarray]] = None,
          stddev: Optional[Union[float, np.ndarray]] = None):
    """
    Evaluates the Gaussian function over a specified domain in multiple
    dimensions.
    If a domain is specified, the function will evaluate the Gaussian at
    linearly interpolated values between the lower and upper bounds of the
    domain. The number of samples is determined by the shape parameter. For
    example, for given parameters shape=(5,) and domain=[[-5, -1]], it will
    evaluate the function at positions -5,-4,-3,-2,-1.
    If no domain is specified, the function will evaluate the Gaussian at the
    indices of the sampling points. For instance, for a given shape of
    shape=(5,), it will evaluate the function at positions 0,1,2,3,4.
    :param shape: number of sampling points along each dimension
    :type shape: tuple(int)
    :param domain: lower and upper bound of
        input values for each dimension at which the Gaussian function is
        evaluated
    :type domain: numpy.ndarray, optional
    :param amplitude: amplitude of the Gaussian
    :type amplitude: float
    :param mean: mean of the Gaussian
    :type mean: numpy.ndarray
    :param stddev: standard deviation of the Gaussian
    :type stddev: numpy.ndarray
    :return: multi-dimensional array with samples of the Gaussian
    :rtype: numpy.ndarray
    """
    # dimensionality of the Gaussian
    dimensionality = len(shape)

    # if no domain is specified, use the indices of
    # the sampling points as domain
    if domain is None:
        domain = np.zeros((dimensionality, 2))
        domain[:, 1] = np.array(shape[:]) - 1

    # standard deviation is one by default
    if stddev is None:
        stddev = np.ones(shape)

    # create linear spaces for each dimension
    linspaces = []
    for i in range(dimensionality):
        linspaces.append(np.linspace(domain[i, 0], domain[i, 1], shape[i]))

    # arrange linear spaces into a meshgrid
    linspaces = np.array(linspaces, dtype=object)
    grid = np.meshgrid(*linspaces, copy=False)
    grid = np.array(grid)

    # swap axes to get around perculiarity of meshgrid
    if dimensionality > 1:
        grid = np.swapaxes(grid, 1, 2)

    # reshape meshgrid to fit multi-variate normal
    grid = np.moveaxis(grid, 0, -1)

    # compute normal probability density function
    mv_normal_pdf = scipy.stats.multivariate_normal.pdf(grid,
                                                        mean=mean,
                                                        cov=stddev)
    # normalize probability density function and apply amplitude
    gaussian = amplitude * mv_normal_pdf / np.max(mv_normal_pdf)

    return gaussian


class InputPattern(ABC):
    def __init__(self, function, **kwargs):
        self._function = function
        self._pattern_params = kwargs

    def __call__(self, shape: ty.Tuple[int]) -> np.ndarray:
        return self._function(shape, **self._pattern_params)


class GaussInputPattern(InputPattern):
    def __init__(self, **kwargs):
        self._validate_params(**kwargs)
        super().__init__(gauss, **kwargs)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self._pattern_params[key] = value

        self._validate_params(**self._pattern_params)

    def _validate_params(self,
                         amplitude: float,
                         mean: Union[float, np.ndarray],
                         stddev: Union[float, np.ndarray]):
        return


class AbstractInputGenerator(ABC):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 input_pattern: InputPattern):
        self._shape = shape
        self._input_pattern = input_pattern

    @abstractmethod
    def generate(self, *args) -> np.ndarray:
        pass


class SpikeInputGenerator(AbstractInputGenerator):
    def generate(self,
                 time_step: int) -> np.ndarray:
        spike_rates = self._input_pattern(self._shape)
        return self._to_spikes(spike_rates, time_step)

    def _to_spikes(self,
                   spike_rates: np.ndarray,
                   time_step: int) -> np.ndarray:
        pass  # TODO


class BiasInputGenerator(AbstractInputGenerator):
    def generate(self) -> np.ndarray:
        return self._input_pattern(self._shape)

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
    def __init__(self, shape):
        self._shape = shape
        self._pattern = None

    @property
    def shape(self):
        return self._shape

    @property
    def pattern(self) -> np.ndarray:
        return self._pattern

    @abstractmethod
    def _update(self):
        pass


class GaussInputPattern(InputPattern):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 amplitude: float,
                 mean: ty.Union[float, np.ndarray],
                 stddev: ty.Union[float, np.ndarray]):
        super().__init__(shape)

        amplitude, mean, stddev = self._validate_gauss_params(amplitude, mean, stddev)

        self._amplitude = amplitude
        self._mean = mean
        self._stddev = stddev

        self._update()

    def _update(self):
        self._pattern = gauss(self._shape,
                              amplitude=self._amplitude,
                              mean=self._mean,
                              stddev=self._stddev)

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @property
    def mean(self) -> ty.Union[float, np.ndarray]:
        return self._mean

    @property
    def stddev(self) -> ty.Union[float, np.ndarray]:
        return self._stddev

    @amplitude.setter
    def amplitude(self, amplitude: float):
        amplitude, _, _ = self._validate_gauss_params(amplitude, self._mean, self._stddev)
        self._amplitude = amplitude
        self._update()

    @mean.setter
    def mean(self, mean: float):
        _, mean, _ = self._validate_gauss_params(self._amplitude, mean, self._stddev)
        self._mean = mean
        self._update()

    @stddev.setter
    def stddev(self, stddev: float):
        _, _, stddev = self._validate_gauss_params(self._amplitude, self._mean, stddev)
        self._stddev = stddev
        self._update()

    def _validate_gauss_params(self,
                               amplitude: float,
                               mean: ty.Union[float, ty.List[float]],
                               stddev: ty.Union[float, ty.List[float]]) \
            -> ty.Tuple[float, ty.List[float], ty.List[float]]:
        if isinstance(mean, float) or isinstance(mean, int):
            mean = float(mean)
            mean = [mean]

        if isinstance(stddev, float) or isinstance(stddev, int):
            stddev = float(stddev)
            stddev = [stddev]

        if len(mean) == 1 and len(stddev) == 1:
            mean_val = mean[0]
            stddev_val = stddev[0]
            mean = [mean_val for _ in range(len(self.shape))]
            stddev = [stddev_val for _ in range(len(self.shape))]

        if len(mean) > 1 and len(stddev) == 1:
            if len(mean) != len(self.shape):
                raise ValueError("<mean> parameter has length different from shape dimensionality")

            stddev_val = stddev[0]
            stddev = [stddev_val for _ in range(len(self.shape))]

        if len(mean) == 1 and len(stddev) > 1:
            if len(stddev) != len(self.shape):
                raise ValueError("<stddev> parameter has length different from shape dimensionality")

            mean_val = mean[0]
            mean = [mean_val for _ in range(len(self.shape))]

        if len(mean) > 1 and len(stddev) > 1:
            if len(mean) != len(self.shape):
                raise ValueError("<mean> parameter has length different from shape dimensionality")
            if len(stddev) != len(self.shape):
                raise ValueError("<stddev> parameter has length different from shape dimensionality")

        return amplitude, mean, stddev


class AbstractInputGenerator(ABC):
    def __init__(self,
                 input_pattern: InputPattern):
        self._input_pattern = input_pattern

    @property
    def shape(self):
        return self._input_pattern.shape

    @abstractmethod
    def generate(self, *args) -> np.ndarray:
        pass


TIME_STEPS_PER_MINUTE = 6000.0


class SpikeInputGenerator(AbstractInputGenerator):
    def __init__(self,
                 input_pattern: InputPattern):
        super().__init__(input_pattern)

        # Unit = Number of time steps
        self._inter_spike_distances = self._compute_distances(self._input_pattern.pattern)

        self._last_spiked_time_step = np.full(self._input_pattern.shape, - np.inf)

        print(self._inter_spike_distances)
        print(self._last_spiked_time_step)

    def generate(self,
                 time_step: int) -> np.ndarray:
        return self._generate_spikes(time_step)

    def _compute_distances(self, pattern):
        distances = np.zeros_like(pattern)

        for idx, item in enumerate(pattern.flat):
            if item > 0.5:
                distance = np.int(np.rint(TIME_STEPS_PER_MINUTE / item))

                distances.flat[idx] = distance
            else:
                distances.flat[idx] = np.inf

        return distances

    def _generate_spikes(self,
                         time_step: int) -> np.ndarray:
        distances_last_spiked = time_step - self._last_spiked_time_step
        result = distances_last_spiked >= self._inter_spike_distances

        self._last_spiked_time_step[result] = time_step

        return result


class BiasInputGenerator(AbstractInputGenerator):
    def generate(self) -> np.ndarray:
        return self._input_pattern.pattern

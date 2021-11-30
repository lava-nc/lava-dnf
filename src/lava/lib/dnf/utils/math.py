# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import scipy.stats


def gauss(shape: ty.Tuple[int, ...],
          domain: ty.Optional[np.ndarray] = None,
          amplitude: ty.Optional[float] = 1.0,
          mean: ty.Optional[ty.Union[float, np.ndarray]] = None,
          stddev: ty.Optional[ty.Union[float, np.ndarray]] = None
          ) -> np.ndarray:
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

    Parameters
    ----------
    shape : tuple(int)
        number of sampling points along each dimension
    domain : numpy.ndarray, optional
        lower and upper bound of input values for each dimension at which
        the Gaussian function is evaluated
    amplitude : float, optional
        amplitude of the Gaussian, defaults to 1
    mean : numpy.ndarray, optional
        mean of the Gaussian, defaults to 0
    stddev : numpy.ndarray, optional
        standard deviation of the Gaussian, defaults to 1

    Returns
    -------
    gaussian : numpy.ndarray
        multi-dimensional array with samples of the Gaussian
    """
    # Dimensionality of the Gaussian
    dimensionality = len(shape)

    # Domain defaults to the indices of the sampling points
    if domain is None:
        domain = np.zeros((dimensionality, 2))
        domain[:, 1] = np.array(shape[:]) - 1
    else:
        if isinstance(domain, np.ndarray) and domain.shape != (len(shape), 2):
            raise ValueError("the shape of <domain> is incompatible with "
                             "the specified <shape>; <domain> should be of "
                             f"shape ({len(shape)}, 2) but is {domain.shape}")

    # Mean defaults to 0
    if mean is None:
        mean = np.zeros((dimensionality,))
    else:
        if isinstance(mean, np.ndarray) and mean.shape != (len(shape),):
            raise ValueError("the shape of <mean> is incompatible with "
                             "the specified <shape>; <mean> should be of "
                             f"shape ({len(shape)},) but is {mean.shape}")

    # Standard deviation defaults to 1
    if stddev is None:
        stddev = np.ones((dimensionality,))
    else:
        if isinstance(stddev, np.ndarray) and stddev.shape != (len(shape),):
            raise ValueError("the shape of <stddev> is incompatible with "
                             "the specified <shape>; <stddev> should be of "
                             f"shape ({len(shape)},) but is {stddev.shape}")

    # Create linear spaces for each dimension
    linspaces = []
    for i in range(dimensionality):
        linspaces.append(np.linspace(domain[i, 0], domain[i, 1], shape[i]))

    # Arrange linear spaces into a meshgrid
    linspaces = np.array(linspaces, dtype=object)
    grid = np.meshgrid(*linspaces, copy=False)
    grid = np.array(grid)

    # Swap axes to get around peculiarity of meshgrid
    if dimensionality > 1:
        grid = np.swapaxes(grid, 1, 2)

    # Reshape meshgrid to fit multi-variate normal
    grid = np.moveaxis(grid, 0, -1)

    # Compute normal probability density function
    mv_normal_pdf = scipy.stats.multivariate_normal.pdf(grid,
                                                        mean=mean,
                                                        cov=stddev)

    # Add singleton axes that were dropped in creating the pdf
    if mv_normal_pdf.shape != shape:
        for axis, size in enumerate(shape):
            if size == 1:
                mv_normal_pdf = np.expand_dims(mv_normal_pdf, axis=axis)

    # Normalize probability density function and apply amplitude
    gaussian = amplitude * mv_normal_pdf / np.max(mv_normal_pdf)

    return gaussian


def is_odd(n: int) -> bool:
    """
    Checks whether n is an odd number.

    :param int n: number to check
    :returns bool: True if <n> is an odd number"""
    return bool(n & 1)

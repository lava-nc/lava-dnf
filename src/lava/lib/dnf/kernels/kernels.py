# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import warnings
from abc import ABC, abstractmethod

from lava.lib.dnf.utils.convenience import to_ndarray
from lava.lib.dnf.utils.math import gauss


class Kernel:
    """
    Represents a kernel that can be used in the Convolution operation.

    Parameters
    ----------
    weights : numpy.ndarray
        weight matrix of the kernel
    padding_value : float, optional
        value that is used to pad the kernel when the Convolution operation
        uses BorderType.PADDED
    """
    def __init__(self,
                 weights: np.ndarray,
                 padding_value: ty.Optional[float] = 0) -> None:
        self._weights = weights
        self._padding_value = padding_value

    @property
    def weights(self) -> np.ndarray:
        """Returns the weights"""
        return self._weights

    @property
    def padding_value(self) -> float:
        """Returns the padding value"""
        return self._padding_value


class GaussianMixin(ABC):
    """
    Mixin for kernels that are generated with the gauss function.

    Parameters
    ----------
    amp_exc : float
        amplitude of the excitatory Gaussian of the kernel
    width_exc : list(float)
        widths of the excitatory Gaussian of the kernel
    limit : float
        determines the size/shape of the kernel such that the weight matrix
        will have the size 2*limit*width_exc; defaults to 1
    shape : tuple(int), optional
        will return the weight with this explicit shape; if used, the limit
        argument will have no effect

    """
    def __init__(self,
                 amp_exc: float,
                 width_exc: ty.Union[float, ty.List[float]],
                 limit: ty.Optional[float] = 1.0,
                 shape: ty.Optional[ty.Tuple[int, ...]] = None,
                 dominant_width: np.ndarray = None) -> None:

        if amp_exc < 0:
            raise ValueError("<amp_exc> must be positive")
        self._amp_exc = amp_exc

        self._width_exc = to_ndarray(width_exc)

        if limit < 0:
            raise ValueError("<limit> must be positive")
        self._limit = limit

        if dominant_width is None:
            dominant_width = self._width_exc

        self._shape = self._compute_shape(dominant_width) \
            if shape is None else self._validate_shape(shape, dominant_width)

    @abstractmethod
    def _compute_weights(self) -> np.ndarray:
        """
        Computes the weights of the kernel

        Returns
        -------
        weights : numpy.ndarray
            computed weights of the kernel

        """
        pass

    def _compute_domain(self) -> np.ndarray:
        """
        Computes the domain of a kernel for computing the Gaussian.

        Returns
        -------
        domain : numpy.ndarray
            domain used for the gauss() function

        """
        shape = np.array(self._shape)
        domain = np.zeros((len(shape), 2))
        half_domain = shape / 2.0
        domain[:, 0] = -half_domain
        domain[:, 1] = half_domain

        return domain

    @staticmethod
    def _validate_shape(shape: ty.Tuple[int, ...],
                        width: np.ndarray) -> ty.Tuple[int, ...]:
        """
        Validates the shape of the kernel against a width parameter

        Parameters
        ----------
        shape : tuple(int)
            shape to be validated
        width : numpy.ndarray
            width to validate the shape against

        Returns
        -------
        shape : tuple(int)
            validated shape

        """
        if np.size(width, axis=0) != len(shape):
            raise ValueError("<width_exc> and <shape> are incompatible; the"
                             "number of entries in <width_exc> must match the"
                             "number of dimensions in <shape>")

        if np.any(np.array(shape)[:] % 2 == 0):
            warnings.warn("kernel has an even size; this may introduce drift")

        return shape

    def _compute_shape(self, width: np.ndarray) -> ty.Tuple[int, ...]:
        """
        Computes the shape of a kernel from the a width-parameter of the
        kernel and a limit factor.

        Parameters
        ----------
        width : numpy.ndarray
            width parameter to determine the shape

        Returns
        -------
        shape : tuple(int)
            shape of the kernel
        """
        # Compute shape from limit
        shape = np.uint(np.ceil(2 * self._limit * width))

        # Ensure that the kernel has an odd size
        shape = np.where(shape % 2 == 0, shape + 1, shape)

        return tuple(shape)


class SelectiveKernel(GaussianMixin, Kernel):
    """
    A kernel that enables creating a selective dynamic neural field
    (local excitation, global inhibition).

    Parameters
    ----------
    amp_exc : float
        amplitude of the excitatory Gaussian of the kernel
    width_exc : list(float)
        widths of the excitatory Gaussian of the kernel
    global_inh : float
        global inhibition of the kernel; must be negative
    limit : float
        determines the size/shape of the kernel such that the weight matrix
        will have the size 2*limit*width_exc; defaults to 1
    shape : tuple(int), optional
        will return the weight with this explicit shape; if used, the limit
        argument will have no effect

    """
    def __init__(self,
                 amp_exc: float,
                 width_exc: ty.Union[float, ty.List[float]],
                 global_inh: float,
                 limit: ty.Optional[float] = 1.0,
                 shape: ty.Optional[ty.Tuple[int, ...]] = None) -> None:

        GaussianMixin.__init__(self, amp_exc, width_exc, limit, shape)

        if global_inh > 0:
            raise ValueError("<global_inh> must be negative")
        self._global_inh = global_inh

        weights = self._compute_weights()
        Kernel.__init__(self, weights=weights, padding_value=self._global_inh)

    def _compute_weights(self) -> np.ndarray:
        local_excitation = gauss(self._shape,
                                 domain=self._compute_domain(),
                                 amplitude=self._amp_exc,
                                 stddev=self._width_exc)

        return local_excitation + self._global_inh


class MultiPeakKernel(GaussianMixin, Kernel):
    """
    "Mexican hat" kernel (local excitation and mid-range inhibition) for a
    DNF that enables it to create multiple peaks.

    Parameters
    ----------
    amp_inh : float
        amplitude of the inhibitory Gaussian of the kernel
    width_inh : list(float)
        widths of the inhibitory Gaussian of the kernel

    """
    def __init__(self,
                 amp_exc: float,
                 width_exc: ty.Union[float, ty.List[float]],
                 amp_inh: float,
                 width_inh: ty.Union[float, ty.List[float]],
                 limit: float = 1.0,
                 shape: ty.Optional[ty.Tuple[int]] = None) -> None:

        if amp_inh > 0:
            raise ValueError("<amp_inh> must be positive")
        self._amp_inh = amp_inh

        self._width_inh = to_ndarray(width_inh)

        GaussianMixin.__init__(self,
                               amp_exc,
                               width_exc,
                               limit,
                               shape,
                               dominant_width=self._width_inh)

        self._validate_widths(self._width_exc, self._width_inh)

        weights = self._compute_weights()
        Kernel.__init__(self, weights=weights)

    @staticmethod
    def _validate_widths(width_exc: np.ndarray,
                         width_inh: np.ndarray) -> None:
        """Validates the excitatory and inhibitory widths against each other."""

        if width_exc.shape != width_inh.shape:
            raise ValueError("shape of <width_exc> "
                             f"{width_exc.shape} != {width_inh.shape} shape of "
                             "<width_inh>")

    def _compute_weights(self) -> np.ndarray:
        domain = self._compute_domain()

        local_excitation = gauss(self._shape,
                                 domain=domain,
                                 amplitude=self._amp_exc,
                                 stddev=self._width_exc)

        mid_range_inhibition = gauss(self._shape,
                                     domain=domain,
                                     amplitude=self._amp_inh,
                                     stddev=self._width_inh)

        return local_excitation + mid_range_inhibition

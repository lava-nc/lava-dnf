# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import warnings

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
                 padding_value: ty.Optional[float] = 0):
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


class SelectiveKernel(Kernel):
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
                 shape: ty.Optional[ty.Tuple[int, ...]] = None):
        super().__init__(weights=np.zeros((1,)))

        if amp_exc < 0:
            raise ValueError("<amp_exc> must be positive")
        self._amp_exc = amp_exc

        self._width_exc = to_ndarray(width_exc)

        if global_inh > 0:
            raise ValueError("<global_inh> must be negative")
        self._global_inh = global_inh

        if limit < 0:
            raise ValueError("<limit> must be positive")
        self._limit = limit

        self._shape = self._compute_shape(self._limit, self._width_exc) \
            if shape is None else self._validate_shape(shape, self._width_exc)

        weights = self._compute_weights()
        super().__init__(weights=weights, padding_value=self._global_inh)

    def _compute_weights(self):
        """Computes the weights of the kernel"""
        local_excitation = gauss(self._shape,
                                 domain=self._compute_domain(self._shape),
                                 amplitude=self._amp_exc,
                                 stddev=self._width_exc,
                                 mean=np.zeros_like(self._width_exc))

        return local_excitation + self._global_inh

    @staticmethod
    def _compute_domain(shape: ty.Tuple[int, ...]):
        """
        Computes the domain of a kernel for computing the Gaussian.

        Parameters
        ----------
        shape : tuple(int)
            shape of the kernel

        Returns
        -------
        domain : numpy.ndarray
            domain used for the gauss() function

        """
        # compute the domain of the kernel
        _shape = np.array(shape)
        domain = np.zeros((len(_shape), 2))
        half_domain = _shape / 2.0
        domain[:, 0] = -half_domain
        domain[:, 1] = half_domain

        return domain

    @staticmethod
    def _validate_shape(shape: ty.Tuple[int, ...],
                        width: np.ndarray):
        """Validate shape of a kernel"""
        if np.size(width, axis=0) != len(shape):
            raise ValueError("<width_exc> and <shape> are incompatible; the"
                             "number of entries in <width_exc> must match the"
                             "number of dimensions in <shape>")

        if np.any(np.array(shape)[:] % 2 == 0):
            warnings.warn("kernel has an even size; this may introduce drift")

        return shape

    @staticmethod
    def _compute_shape(limit: float, width: np.ndarray):
        """
        Computes the shape of a kernel from the a width-parameter of the
        kernel and a limit factor.

        Parameters
        ----------
        limit : float
            factor that controls the shape

        width : numpy.ndarray
            parameter that determines the width of the

        Returns
        -------
        shape : tuple(int)
            shape of the kernel

        """

        # compute shape from limit
        shape = np.uint(np.ceil(2 * limit * width))

        # ensure that the kernel has an odd size
        shape = np.where(shape % 2 == 0, shape + 1, shape)

        return tuple(shape)

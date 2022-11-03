# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from dv import AedatFile
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class DVSFileInput(AbstractProcess):
    def __init__(self,
                 true_height: int,
                 true_width: int,
                 file_path: str,
                 flatten: bool = False,
                 down_sample_factor: int = 1,
                 down_sample_mode: str = "down_sample",
                 num_steps=1) -> None:
        super().__init__(true_height=true_height,
                         true_width=true_width,
                         file_path=file_path,
                         flatten=flatten,
                         down_sample_factor=down_sample_factor,
                         down_sample_mode=down_sample_mode,
                         num_steps=num_steps)

        down_sampled_height = true_height // down_sample_factor
        down_sampled_width = true_width // down_sample_factor

        if flatten:
            out_shape = (down_sampled_width * down_sampled_height,)
        else:
            out_shape = (down_sampled_width, down_sampled_height)
        self.event_frame_out = OutPort(shape=out_shape)


@implements(proc=DVSFileInput, protocol=LoihiProtocol)
@requires(CPU)
class PyDVSFileInputPM(PyLoihiProcessModel):
    event_frame_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._true_height = proc_params["true_height"]
        self._true_width = proc_params["true_width"]
        self._true_shape = (self._true_width, self._true_height)
        self._file_path = proc_params["file_path"]
        self._aedat_file = AedatFile(self._file_path)
        self._event_stream = self._aedat_file["events"].numpy()
        self._frame_stream = self._aedat_file["frames"]
        self._flatten = proc_params["flatten"]
        self._down_sample_factor = proc_params["down_sample_factor"]
        self._down_sample_mode = proc_params["down_sample_mode"]
        self._down_sampled_height = \
            self._true_height // self._down_sample_factor
        self._down_sampled_width = \
            self._true_width // self._down_sample_factor
        self._down_sampled_shape = (self._down_sampled_width,
                                    self._down_sampled_height)
        self._num_steps = proc_params["num_steps"]

    def run_spk(self):
        events = self._event_stream.__next__()
        xs, ys, ps = events['x'], events['y'], events['polarity']

        event_frame = np.zeros(self._true_shape)
        event_frame[xs[ps == 0], ys[ps == 0]] = 1
        event_frame[xs[ps == 1], ys[ps == 1]] = 1

        if self._down_sample_mode == "down_sampling":
            event_frame_small = \
                event_frame[::self._down_sample_factor,
                ::self._down_sample_factor]

            event_frame_small = \
                event_frame_small[:self._down_sampled_height,
                :self._down_sampled_width]
        elif self._down_sample_mode == "max_pooling":
            event_frame_small = \
                self._pool_2d(event_frame, kernel_size=self._down_sample_factor,
                              stride=self._down_sample_factor, padding=0,
                              pool_mode='max')
        elif self._down_sample_mode == "convolution":
            event_frame_small = \
                self._convolution(event_frame)
        else:
            raise ValueError(f"Unknown down_sample_mode "
                             f"{self._down_sample_mode}")

        if self._flatten:
            event_frame_small = event_frame_small.flatten()
        self.event_frame_out.send(event_frame_small)

    def _pool_2d(self, matrix: np.ndarray, kernel_size: int, stride: int,
                 padding: int = 0, pool_mode: str = 'max'):
        # Padding
        padded_matrix = np.pad(matrix, padding, mode='constant')

        # Window view of A
        output_shape = ((padded_matrix.shape[0] - kernel_size) // stride + 1,
                        (padded_matrix.shape[1] - kernel_size) // stride + 1)
        shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
        strides_w = (stride * padded_matrix.strides[0],
                     stride * padded_matrix.strides[1],
                     padded_matrix.strides[0],
                     padded_matrix.strides[1])
        matrix_w = as_strided(padded_matrix, shape_w, strides_w)

        # Return the result of pooling
        if pool_mode == 'max':
            return matrix_w.max(axis=(2, 3))
        elif pool_mode == 'avg':
            return matrix_w.mean(axis=(2, 3))

    def _convolution(self, matrix: np.ndarray, kernel_size: int = 3):
        kernel = np.ones((kernel_size, kernel_size))
        event_frame_convolved = signal.convolve2d(matrix, kernel, mode="same")

        event_frame_small = \
            event_frame_convolved[::self._down_sample_factor,
            ::self._down_sample_factor]

        event_frame_small = \
            event_frame_small[:self._down_sampled_width,
            :self._down_sampled_height]

        return event_frame_small

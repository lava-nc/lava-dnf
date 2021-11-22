# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.proc.dense.models import Dense
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.lib.dnf.operations.operations import AbstractOperation
from lava.lib.dnf.connect.exceptions import MisconfiguredConnectError
from lava.lib.dnf.connect.reshape_bool.process import ReshapeBool
from lava.lib.dnf.connect.reshape_int.process import ReshapeInt


def connect(
    src_op: OutPort,
    dst_ip: InPort,
    ops: ty.Union[ty.List[AbstractOperation], AbstractOperation]
) -> AbstractProcess:
    """
    Creates and returns a Connections Process <conn> and connects the source
    OutPort <src_op> to the InPort of <conn> and the OutPort of <conn> to the
    InPort of <dst_ip>.

    The connectivity is generated from a list of operation objects <ops>.
    Each operation generates a dense connectivity matrix based
    on its parameters. These matrices are multiplied into a single
    connectivity matrix, which is then used to generate a Connections Process
    between source and destination.

    Parameters
    ----------
    src_op : OutPort
        OutPort of the source Process that will be connected
    dst_ip : InPort
        InPort of the destination Process that will be connected
    ops : list(AbstractOperation)
        list of operations that describes how the connection between
        <src_op> and <dst_ip> will be created

    Returns
    -------
    connections : AbstractProcess
        process containing the connections between <src_op> and <dst_ip>

    """
    # validate the list of operations
    ops = _validate_ops(ops)

    # configure all operations in the <ops> list with input and output shape
    _configure_ops(ops, src_op.shape, dst_ip.shape)

    # compute the connectivity matrix of all operations and multiply them
    # into a single matrix <weights> that will be used for the Process
    weights = _compute_weights(ops)

    # create Connections process and connect it:
    # source -> connections -> destination
    connections = _make_connections(src_op, dst_ip, weights)

    return connections


def _configure_ops(
    ops: ty.List[AbstractOperation],
    src_shape: ty.Tuple[int, ...],
    dst_shape: ty.Tuple[int, ...]
):
    """
    Configure all operations by setting their input and output shape and
    checking that the final output shape matches the shape of the destination
    InPort.

    Parameters
    ----------
    ops : list(AbstractOperation)
        list of operations to configure
    src_shape : tuple(int)
        shape of the OutPort of the source Process
    dst_shape : tuple(int)
        shape of the InPort of the destination Process

    Returns
    -------

    """
    # we go from the source through all operations and memorize the output
    # shape of the last operation (here, the source)
    prev_output_shape = src_shape

    # for every operation in the list of operations
    for op in ops:
        # let the operation configure the output shape given the incoming
        # input shape
        input_shape = prev_output_shape
        op.configure(input_shape)
        # memorize the computed output shape for the next iteration of the loop
        prev_output_shape = op.output_shape

    # check that the output shape of the last operation matches the shape of
    # the InPort of the destination Process
    if prev_output_shape != dst_shape:
        raise MisconfiguredConnectError(
            "the output shape of the last operation does not match the shape "
            "of the destination InPort; some operations may be misconfigured")


def _validate_ops(
    ops: ty.Union[AbstractOperation, ty.List[AbstractOperation]]
) -> ty.List[AbstractOperation]:
    """
    Validates the <ops> argument of the 'connect' function

    Parameters:
    -----------
    ops : list or AbstractOperation
        the list of operations to be validated

    Returns:
    --------
    ops : list
        validated list of operations

    """
    # make <ops> a list if it is not one already
    if not isinstance(ops, list):
        ops = [ops]

    # empty lists raise an error
    if len(ops) == 0:
        raise ValueError("list of operations is empty")

    # check whether each element in <operations> is of type AbstractOperation
    for op in ops:
        if not isinstance(op, AbstractOperation):
            raise TypeError("elements in list of operations must be of type "
                            f"Operation, found type {type(op)}")

    return ops


def _compute_weights(ops: ty.List[AbstractOperation]) -> np.ndarray:
    """
    Compute the overall connectivity matrix to be used for the Connections
    Process from the individual connectivity matrices that each operation
    produces.

    Parameters
    ----------
    ops : list(AbstractOperation)
        list of operations

    Returns
    -------
    weights : np.ndarray

    """
    weights = None

    # for every operation
    for op in ops:
        # compute the weights of the current operation
        op_weights = op.compute_weights()

        # if it's the first operation in the list, initialize overall
        # weights
        if weights is None:
            weights = op_weights
        # otherwise multiply weights with the connectivity matrix from the last
        # operations in the list to create the overall weights matrix
        else:
            weights = np.matmul(op_weights, weights)

    return weights


def _make_connections(src_op: OutPort,
                      dst_ip: InPort,
                      weights: np.ndarray) -> AbstractProcess:
    """
    Creates a Connections Process with the given weights and connects its
    ports such that:
    source-OutPort -> connections-InPort and
    connections-InPort -> destination-OutPort

    Parameters
    ----------
    src_op : OutPort
        OutPort of the source Process
    dst_ip : InPort
        InPort of the destination Process
    weights : numpy.ndarray
        connectivity weight matrix used for the Connections Process

    Returns
    -------
    Connections Process : AbstractProcess

    """

    # create the connections process
    connections = Dense(shape=weights.shape,
                        weights=weights)

    # make connections from the source port to the connections process
    # TODO (MR) workaround in absence of ReshapePorts
    con_ip = connections.s_in
    rs1 = ReshapeBool(shape_in=src_op.shape, shape_out=con_ip.shape)
    src_op.connect(rs1.s_in)
    rs1.s_out.connect(con_ip)

    # make connections from the connections process to the destination port
    # TODO (MR) workaround in absence of ReshapePorts
    con_op = connections.a_out
    rs2 = ReshapeInt(shape_in=con_op.shape, shape_out=dst_ip.shape)
    con_op.connect(rs2.s_in)
    rs2.s_out.connect(dst_ip)

    return connections

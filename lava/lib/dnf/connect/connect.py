# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
import itertools

from lava.magma.core.process.process import AbstractProcess
from lava.proc.dense.models import Dense
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.lib.dnf.operations.operations import AbstractOperation
from lava.lib.dnf.connect.exceptions import MissingOpError, DuplicateOpError
from lava.lib.dnf.utils.convenience import num_dims


def connect(src_op: OutPort,
            dst_ip: InPort,
            ops: ty.List[AbstractOperation]) -> AbstractProcess:
    """
    Creates a Connections Process <conn> and connects the source OutPort
    <src_op> to the InPort of <conn> and the OutPort of <conn> to the InPort
    of <dst_ip>.

    The list of operations <ops> is a description of what the Connections
    Process <conn> will look like.

    The connectivity is generated from a list of operation objects <ops>.
    Each operation generates a dense connectivity matrix based
    on its parameters. These matrices are multiplied into a single
    connectivity matrix, which is then used to generate a Connections Process
    between source and destination.

    If less than a third of the entries of the connectivity
    matrix are non-zero, the connect() function instantiates a
    'lava.proc.Sparse' Process; otherwise it instantiates a 'lava.proc.Dense'
    process.

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
    # validate the list of operations, including validation against the shapes
    # of the source and destination ports
    ops = validate_ops(ops, src_op.shape, dst_ip.shape)

    # configure all operations in the <ops> list with input and output shape
    configure_ops(ops, src_op.shape, dst_ip.shape)

    # compute the connectivity matrix of each operation and multiply them
    # into a single matrix <weights> that will be used for the Process
    weights = compute_weights(ops)

    # create connections process and connect it:
    # source -> connections -> destination
    connections = make_connections(src_op, dst_ip, weights)

    return connections


def configure_ops(
        ops: ty.List[AbstractOperation],
        src_shape: ty.Tuple[int, ...],
        dst_shape: ty.Tuple[int, ...]
):
    # we go from the source through all operations and memorize the output
    # shape of the last operation (here, the source)
    prev_output_shape = src_shape

    # for every operation in the list of operations
    for op in ops:
        # set the input of the current operation to the incoming shape
        # (also, initialize output shape to the same value)
        input_shape = output_shape = prev_output_shape

        # if the current operation changes the shape ...
        if op.changes_dim or op.changes_size or op.reorders_shape:
            # ... its output shape to the shape of the destination ...
            output_shape = dst_shape
            # ... then update <prev_output_shape>
            prev_output_shape = output_shape

        # configure the current operation with input and output shape
        op.configure(input_shape, output_shape)


def validate_ops(
        ops: ty.Union[AbstractOperation, ty.List[AbstractOperation]],
        src_shape: ty.Tuple[int, ...],
        dst_shape: ty.Tuple[int, ...]
) -> ty.List[AbstractOperation]:
    """
    Validates the <ops> argument of the 'connect' function

    Parameters:
    -----------
    ops : list or AbstractOperation
        the list of operations to be validated
    src_shape : tuple(int)
        shape of the OutPort of the source Process
    dst_shape : tuple(int)
        shape of the InPort of the destination Process

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

    # strings that represent the three types of operations
    changes_dim = "changes_dim"
    changes_size = "changes_size"
    reorders_shape = "reorders_shape"

    # if the shape of the source OutPort differs from the destination InPort ...
    if src_shape != dst_shape:
        # ... and the shapes differ in dimensionality ...
        if num_dims(src_shape) != num_dims(dst_shape):
            # ... check whether an operation has been specified that
            # changes the dimensionality.
            if not np.any([op.changes_dim for op in ops]):
                raise MissingOpError(changes_dim)
        # ... and the shapes have the same length ...
        else:
            # ... if the shapes can just be reordered ...
            if dst_shape in list(itertools.permutations(src_shape)):
                # ... check whether an operation has been specified that
                # reorders the shape.
                if not np.any([op.reorders_shape for op in ops]):
                    raise MissingOpError(reorders_shape)
            # ... if there is an actual difference in size ...
            else:
                # ... check whether an operation has been specified that
                # resizes the shape.
                if not np.any([op.changes_size for op in ops]):
                    raise MissingOpError(changes_size)

    # we count the number of instances of every type of operation in this
    # dict to raise errors when multiple instances of the same type are
    # specified that would not work
    op_type_counter = {changes_dim: 0,
                       changes_size: 0,
                       reorders_shape: 0}

    # we also keep track of whether multiple operations are set up to change
    # the shape, which is currently not implemented
    multiple_ops_change_shape = False
    prev_op_changes_shape = False

    for op in ops:
        op_changes_shape = False

        # check whether each element in <operations> is of type Operation
        if not isinstance(op, AbstractOperation):
            raise TypeError("elements in list of operations must be of type"
                            f"Operation, found type {type(op)}")

        else:
            # count the number of instances of every type of operation
            if op.changes_dim:
                op_type_counter[changes_dim] += 1
                op_changes_shape = True
            if op.changes_size:
                op_type_counter[changes_size] += 1
                op_changes_shape = True
            if op.reorders_shape:
                op_type_counter[reorders_shape] += 1
                op_changes_shape = True

        if op_changes_shape:
            if prev_op_changes_shape:
                multiple_ops_change_shape = True
            else:
                prev_op_changes_shape = op_changes_shape

    # raise an exception if multiple operations change dimensionality
    if op_type_counter[changes_dim] > 1:
        raise DuplicateOpError(changes_dim)

    # raise an exception if multiple operations change size
    if op_type_counter[changes_size] > 1:
        raise DuplicateOpError(changes_size)

    # raise an exception if multiple operations reorder the shape
    if op_type_counter[reorders_shape] > 1:
        raise DuplicateOpError(reorders_shape)

    # raise an exception if multiple operations change the shape
    if multiple_ops_change_shape:
        raise NotImplementedError("specifying multiple operations that "
                                  "change the shape is currently not "
                                  "supported")

    return ops


def compute_weights(ops: ty.List[AbstractOperation]) -> np.ndarray:
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
    # initialize overall weights
    weights = None

    # for every operation
    for op in ops:
        # compute the weights of the current operation ...
        op_weights = op.compute_weights()

        if weights is None:
            weights = op_weights
        # ... and multiply it with the connectivity matrix from the last
        # operations in the list to create the overall weights matrix
        else:
            weights = np.matmul(op_weights, weights)

    return weights


def make_connections(src_op: OutPort,
                     dst_ip: InPort,
                     weights: np.ndarray) -> AbstractProcess:
    """
    Creates a Connections Process with the given weights and connects its
    ports such that
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
    src_op_flat = src_op.reshape(connections.in_ports.s_in.shape)
    src_op_flat.connect(connections.in_ports.s_in)
    # make connections from the connections process to the destination port
    con_op = connections.out_ports.a_out.reshape(dst_ip.shape)
    con_op.connect(dst_ip)

    return connections

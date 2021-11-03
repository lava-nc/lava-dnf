# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
import itertools

from lava.magma.core.process.process import AbstractProcess
from lava.proc.dense.models import Dense
from lava.magma.core.process.ports.ports import InPort, OutPort

from lib.dnf.operations.operations import AbstractOperation
from lib.dnf.connect.exceptions import MissingOpError, DuplicateOpError


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
    ops = validate_ops(ops,
                       src_op.shape,
                       dst_ip.shape)

    weights = ops[0].compute_weights()
    connections = Dense(shape=weights.shape,
                        weights=weights)

    # make connections from the source port to the connections process
    src_op_flat = src_op.reshape(connections.in_ports.s_in.shape)
    src_op_flat.connect(connections.in_ports.s_in)
    # make connections from the connections process to the destination port
    con_op = connections.out_ports.a_out.reshape(dst_ip.shape)
    con_op.connect(dst_ip)

    return connections


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

    # if the shape of the source OutPort differs from the destination InPort ...
    if src_shape != dst_shape:
        # ... and the shapes differ in length ...
        if len(src_shape) != len(dst_shape):
            # ... check whether a projection has been specified.
            if not np.any([op.changes_dim for op in ops]):
                raise MissingOpError("changes_dim=True")
        # ... and the shapes have the same length ...
        else:
            # ... if the shapes can just be reordered ...
            if dst_shape in list(itertools.permutations(src_shape)):
                # ... check whether a Reorder operation has been specified.
                if not np.any([op.reorders_shape for op in ops]):
                    raise MissingOpError("reorders_shape=True")
            # ... if there is an actual difference in size ...
            else:
                # ... check whether a Resize operation has been specified
                if not np.any([op.changes_size for op in ops]):
                    raise MissingOpError("changes_size=True")

    # we count the number of instances of every type of operation in this
    # dict to raise errors when multiple instances of the same type are
    # specified that would not work
    changes_dim = "changes_dim"
    changes_size = "changes_size"
    reorders_shape = "reorders_shape"
    op_type_counter = {changes_dim: 0,
                       changes_size: 0,
                       reorders_shape: 0}

    for op in ops:
        # check whether each element in <operations> is of type Operation
        if not isinstance(op, AbstractOperation):
            raise TypeError("elements in list of operations must be of type"
                            f"Operation, found type {type(op)}")

        else:
            # count the number of instances of every type of operation
            if op.changes_dim:
                op_type_counter[changes_dim] += 1
            if op.changes_size:
                op_type_counter[changes_size] += 1
            if op.reorders_shape:
                op_type_counter[reorders_shape] += 1

    # raise an exception if multiple operations change dimensionality
    if op_type_counter[changes_dim] > 1:
        raise DuplicateOpError("changes_dim=True")

    # raise an exception if multiple operations change size
    if op_type_counter[changes_size] > 1:
        raise DuplicateOpError("changes_size=True")

    # raise an exception if multiple operations reorder the shape
    if op_type_counter[reorders_shape] > 1:
        raise DuplicateOpError("reorders_shape=True")

    return ops

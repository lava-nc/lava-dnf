// INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
//
// Copyright © 2021-2022 Intel Corporation.
//
// This software and the related documents are Intel copyrighted
// materials, and your use of them is governed by the express
// license under which they were provided to you (License). Unless
// the License provides otherwise, you may not use, modify, copy,
// publish, distribute, disclose or transmit  this software or the
// related documents without Intel's prior written permission.
//
// This software and the related documents are provided as is, with
// no express or implied warranties, other than those that are
// expressly stated in the License.
// See: https://spdx.org/licenses/

#include "injector.h"
#include "predefs_InjectorModel.h"

int spk_guard(runState *s)
{
    return 1;
}

void run_spk(runState *rs)
{
    printf("injector beginning");
    int data[15] = {0};
    recv_vec_dense(rs, &in_port, data);
    printf("injector after recv");
    send_vec_dense(rs, &out_port, data);
    printf("injector after send");
}

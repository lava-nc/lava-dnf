// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "injector.h"
#include "predefs_InjectorModel.h"

int spk_guard(runState *s) {
    return 1;
}

void run_spk(runState *rs) {
    int data[10] = {0};
    recv_vec_dense(rs, &in_port, data);
    send_vec_dense(rs, &out_port, data);
}

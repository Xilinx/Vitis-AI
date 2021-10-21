/*
 * Copyright 2020 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// L3
#include "xf_database/gqe_init.hpp"
#include "xf_database/gqe_base.hpp"

namespace xf {
namespace database {
namespace gqe {

// constructor
Base::Base(FpgaInit& obj) {
    ctx = obj.ctx;
    cq = obj.cq;
    prg = obj.prg;

    hbuf_ddr0 = obj.hbuf_ddr0;
    hbuf_ddr1 = obj.hbuf_ddr1;
    for (int i = 0; i < PU_NM; i++) {
        hbuf_hbm[i] = obj.hbuf_hbm[i];
    }

    dbuf_ddr0 = obj.dbuf_ddr0;
    dbuf_ddr1 = obj.dbuf_ddr1;

    for (int i = 0; i < 2 * PU_NM; i++) {
        dbuf_hbm[i] = obj.dbuf_hbm[i];
    }

    buf_accu_ddr0_size = 0;
    buf_accu_ddr1_size = 0;

    buf_idx0 = 0;
    buf_idx1 = 0;
}

// deconstructor
Base::~Base() {}

// set the maximum num of buffers suported to request for ddr0/1
void Base::SetBufAllocMaxNum(int _num) {
    std::cout << "Setting the maximum supported allcation number of host/device buffer to " << _num << std::endl;
    buf_head.resize(2);
    buf_size.resize(2);

    for (int i = 0; i < 2; i++) {
        buf_head[i].resize(_num);
        buf_size[i].resize(_num);
    }
}

// request one host buffer, the size and starting pointer of the buf is recorded
char* Base::AllocHostBuf(bool _ddr_idx, int64_t _size) {
    // align the allocated DDR size to 4KB
    int64_t _size_align = (_size + 4095) / 4096 * 4096;
    int64_t offset = 0;

    if (_ddr_idx == 0) { // ddr0
        buf_head[0][buf_idx0] = buf_accu_ddr0_size;
        buf_size[0][buf_idx0] = _size_align;
        offset = buf_accu_ddr0_size;
        buf_accu_ddr0_size += _size_align;
        buf_idx0++;
        return hbuf_ddr0 + offset;
    } else {
        buf_head[1][buf_idx1] = buf_accu_ddr1_size;
        buf_size[1][buf_idx1] = _size_align;
        offset = buf_accu_ddr1_size;
        buf_accu_ddr1_size += _size_align;
        buf_idx1++;
        return hbuf_ddr1 + offset;
    }
}

} // database
} // gqe
} // xf

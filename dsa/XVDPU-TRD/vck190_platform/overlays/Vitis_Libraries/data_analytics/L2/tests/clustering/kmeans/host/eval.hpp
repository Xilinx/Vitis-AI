/*
 * Copyright 2019 Xilinx, Inc.
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
#ifndef __EVALUE_HPP_
#define __EVALUE_HPP_

#include <sys/time.h>
#include <new>
#include <cstdlib>
#include "../kernel/config.hpp"

#include <ap_int.h>
#include <iostream>

template <typename DT, int sz>
union conv {};
template <>
union conv<double, 64> {
    double dt;
    unsigned long long ut;
};
void convert2array(int dim, int num, ap_uint<512>* data, DType** v) {
    const int sz = sizeof(DType) * 8;
    int p = 0;
    const int sn = 512 / sz;
    const int batch = (dim + sn - 1) / sn;
    int count = num * batch;
    int k = 0;
    int d = 0;
    while (p < count) {
        ap_uint<512> temp = data[p++];
        for (int i = 0; i < 512 / sz; ++i) {
            ap_uint<sz> ut = temp.range((i + 1) * sz - 1, i * sz);
            conv<DType, sz> nt;
            nt.ut = ut;
            v[k][d] = nt.dt;
            d++;
            if (d == dim) {
                k++;
                d = 0;
                break;
            }
        }
    } // while
}

void combineConfig(ap_uint<512>& config, int kcluster, int dim, int nsample, int maxIter, DType eps) {
    config.range(31, 0) = kcluster;
    config.range(63, 32) = dim;
    config.range(95, 64) = nsample;
    config.range(127, 96) = maxIter;
    const int sz = sizeof(DType) * 8;
    conv<DType, sz> nt;
    nt.dt = eps;
    config.range(sz + 127, 128) = nt.ut;
}
void convertVect2axi(DType** v, int dim, int num, int start, ap_uint<512>* data) {
    const int sz = sizeof(DType) * 8;
    ap_uint<1024> baseBlock = 0;
    int offset = 0;
    int q = start;
    int p = 0;
    ap_uint<512> temp = 0;
    for (int k = 0; k < num; ++k) {
        for (int d = 0; d < dim; ++d) {
            DType t = v[k][d];
            conv<DType, sz> nt;
            nt.dt = t;
            temp.range((p + 1) * sz - 1, p * sz) = nt.ut;
            p++;
            if (offset + p * sz >= 512) {
                baseBlock.range(offset + p * sz - 1, offset) = temp.range(p * sz - 1, 0);
                data[q++] = baseBlock.range(511, 0);
                offset += p * sz;
                offset -= 512;
                baseBlock >>= 512;
                p = 0;
                temp = 0;
            }
        }
    } // for
    if (offset > 0 || p > 0) {
        if (p > 0) {
            baseBlock.range(offset + p * sz - 1, offset) = temp.range(p * sz - 1, 0);

            offset += p * sz;
        }
        data[q++] = baseBlock.range(offset, 0);
    }
}
void clear(DType** x, int dim, int kc) {
    for (int ic = 0; ic < kc; ++ic) {
        for (int d = 0; d < dim; ++d) {
            x[ic][d] = 0;
        }
    }
}
void copy(DType** x, DType** y, int dim, int kc) {
    for (int ic = 0; ic < kc; ++ic) {
        for (int d = 0; d < dim; ++d) {
            y[ic][d] = x[ic][d];
        }
    }
}
bool calE(DType** c, DType** nc, int dim, int kc, DType eps) {
    bool res = true;
    for (int ic = 0; ic < kc; ++ic) {
        DType e = 0;
        for (int d = 0; d < dim; ++d) {
            DType df = nc[ic][d] - c[ic][d];
            e += df * df;
        }
        res &= e <= eps;
    }
    return res;
}
void train(DType** s, DType** c, DType** nc, int dim, int kc, int num) {
    int* cnt = (int*)malloc(sizeof(int) * kc);
    for (int ic = 0; ic < kc; ++ic) cnt[ic] = 0;

    for (int id = 0; id < num; ++id) {
        int bk = -1;
        DType md = 1e38;
        for (int ic = 0; ic < kc; ++ic) {
            DType ds = 0;
            for (int d = 0; d < dim; ++d) {
                DType df = (s[id][d] - c[ic][d]);
                ds += df * df;
            }
            if (md > ds) {
                md = ds;
                bk = ic;
            }
        }
        for (int d = 0; d < dim; ++d) {
            nc[bk][d] += s[id][d];
        }
        cnt[bk]++;
    }

    for (int ic = 0; ic < kc; ++ic) {
        int c = cnt[ic] > 0 ? cnt[ic] : 1;
        for (int d = 0; d < dim; ++d) {
            nc[ic][d] /= c;
        }
        std::cout << "cnt[" << ic << "]=" << cnt[ic] << "  " << nc[ic][0] << std::endl;
    }
    free(cnt);
}
void goldenTrain(DType** s, DType** c, DType** nc, int dim, int kc, int num, ap_uint<32>* tag, DType e, int maxIt) {
    int it = 0;
    bool stop;
    do {
        clear(nc, dim, kc);
        train(s, c, nc, dim, kc, num);
        stop = calE(nc, c, dim, kc, e);
        copy(nc, c, dim, kc);
        it++;
    } while (!stop && it < maxIt);
}

#endif //__EVALUE_HPP_

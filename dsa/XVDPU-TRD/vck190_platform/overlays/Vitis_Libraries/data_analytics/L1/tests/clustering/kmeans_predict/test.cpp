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
#include <iostream>
typedef double DType;
#include "config.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#define N 1000  // 999 // 100*100*100*100
#define K (KC)  // 128 // 20 //512 //100 //512 //40 //512 //40 //128 //1024
#define D (DIM) // 512 // 16 //128 //1024 //24 // 512 //80 //128   //1024

#include "xf_data_analytics/clustering/kmeansPredict.hpp"
template <typename DT, int sz>
union conv {
    DT dt;
    unsigned int ut;
};
template <>
union conv<double, 64> {
    double dt;
    unsigned long long ut;
};

void copyToLocal(ap_uint<sizeof(DType) * 8> kcenters[PCU][PDV][UramDepth],
                 ap_uint<sizeof(DType) * 8> centers[PCU][PDV][UramDepth]) {
    for (int i = 0; i < PCU; ++i) {
        for (int j = 0; j < PDV; ++j) {
            for (int k = 0; k < UramDepth; ++k) {
                centers[i][j][k] = kcenters[i][j][k];
            }
        }
    }
}

void copyToLocal(ap_uint<sizeof(DType) * 8> kcenters[PCU][PDV][UramDepth],
                 ap_uint<sizeof(DType) * 8 * PDV> centers[PCU][UramDepth]) {
    const int sz = sizeof(DType) * 8;
    for (int i = 0; i < PCU; ++i) {
        for (int k = 0; k < UramDepth; ++k) {
            ap_uint<sz * PDV> t;
            for (int j = 0; j < PDV; ++j) {
                t.range((j + 1) * sz - 1, j * sz) = kcenters[i][j][k];
            }
            centers[i][k] = t; // kcenters[i][j][k];
        }
    }
}
void kmeansPredict(hls::stream<ap_uint<sizeof(DType) * 8> > sampleStrm[PDV],
                   hls::stream<bool>& endSampleStrm,
                   ap_uint<sizeof(DType) * 8> kcenters[PCU][PDV][UramDepth],
                   const int dims,
                   const int kcluster,
                   hls::stream<ap_uint<32> >& tagStrm,
                   hls::stream<bool>& endTagStrm) {
    // ap_uint<sizeof(DType) * 8> centers[PCU][PDV][UramDepth] = {0};
    ap_uint<sizeof(DType) * 8 * PDV> centers[PCU][UramDepth]; // = {0};
#pragma HLS array_partition variable = centers complete dim = 1
//#pragma HLS array_partition variable = centers complete dim = 2
#pragma HLS bind_storage variable = centers type = ram_2p impl = uram
    copyToLocal(kcenters, centers);
    xf::data_analytics::clustering::kMeansPredict<DType, D, K, UramDepth, PCU, PDV>(
        sampleStrm, endSampleStrm, centers, dims, kcluster, tagStrm, endTagStrm);
}

#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
void convert2Array(DType** v, ap_uint<sizeof(DType) * 8> centers[PCU][PDV][UramDepth], const int dim, const int kc) {
    int batch = (dim + PDV - 1) / PDV;
    for (int i = 0; i < (kc + PCU - 1) / PCU; ++i) {
        for (int j = 0; j < PCU; ++j) {
            for (int k = 0; k < batch; ++k) {
                for (int d = 0; d < PDV; ++d) {
                    int p = i * PCU + j;
                    int q = k * PDV + d;
                    int m = i * batch + k;
                    DType dt = q < dim && p < kc ? v[p][q] : 0;
                    const int sz = sizeof(DType) * 8;
                    conv<DType, sz> nt;
                    nt.dt = dt;
                    centers[j][d][m] = nt.ut;
                }
            }
        }
    }
}

void convert2Strm(DType** v,
                  hls::stream<ap_uint<sizeof(DType) * 8> > sampleStrm[PDV],
                  hls::stream<bool>& endSampleStrm,
                  const int dim,
                  const int num) {
    int batch = (dim + PDV - 1) / PDV;
    for (int i = 0; i < num; ++i) {
        for (int k = 0; k < batch; ++k) {
            for (int d = 0; d < PDV; ++d) {
                int p = i; // i*PCU+j;
                int q = k * PDV + d;
                DType dt = q < dim ? v[p][q] : 0;
                const int sz = sizeof(DType) * 8;
                conv<DType, sz> nt;
                nt.dt = dt;
                sampleStrm[d].write(nt.ut);
            }
            endSampleStrm.write(false);
        }
    }
    endSampleStrm.write(true);
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
    DType e = 0;
    for (int ic = 0; ic < kc; ++ic) {
        for (int d = 0; d < dim; ++d) {
            DType df = nc[ic][d] - c[ic][d];
            e += df * df;
        }
    }
    return e <= eps;
}
// check whether hit another center when both distances are same.
bool checkEqualDis(DType** s, DType** c, int dim, ap_uint<32>* idx, ap_uint<32>* tg1, ap_uint<32>* tg2, int num) {
    bool res = true;
    for (int id = 0; id < num; ++id) {
        DType ds1 = 0;
        DType ds2 = 0;
        ap_uint<32> is = idx[id];
        ap_uint<32> ic1 = tg1[id];
        ap_uint<32> ic2 = tg2[id];
        for (int d = 0; d < dim; ++d) {
            DType df1 = (s[is][d] - c[ic1][d]);
            ds1 += df1 * df1;
            DType df2 = (s[is][d] - c[ic2][d]);
            ds2 += df2 * df2;
        }
        std::cout << "s[" << is << "]=(";
        for (int d = 0; d < dim; ++d) std::cout << " " << s[is][d];
        std::cout << ")" << std::endl;

        std::cout << "c1[" << ic1 << "]=(";
        for (int d = 0; d < dim; ++d) std::cout << " " << c[ic1][d];
        std::cout << ")" << std::endl;

        std::cout << "c2[" << ic2 << "]=(";
        for (int d = 0; d < dim; ++d) std::cout << " " << c[ic2][d];
        std::cout << ")" << std::endl;

        res &= ds1 - ds2 < 1e-8;
        int t = ds1 < ds2 ? ic1 : ic2;
        std::cout << "ds1=" << ds1 << "  ds2=" << ds2 << "    is=" << is << std::endl;
        std::cout << "ic1=" << ic1 << "  ic2=" << ic2 << "    tg=" << t << std::endl;
    }
    return res;
}
void goldenPredict(DType** s, DType** c, int dim, int kc, int num, ap_uint<32>* tag) {
    for (int id = 0; id < num; ++id) {
        ap_uint<32> bk = -1;
        DType md = 1e38;
        for (unsigned int ic = 0; ic < kc; ++ic) {
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
        tag[id] = bk;
    }
}

void train(DType** s, DType** c, DType** nc, int dim, int kc, int num, ap_uint<32>* tag) {
    int* cnt = (int*)malloc(sizeof(int) * kc);
    for (int ic = 0; ic < kc; ++ic) cnt[ic] = 0;

    goldenPredict(s, c, dim, kc, num, tag);
    for (int id = 0; id < num; ++id) {
        ap_uint<32> bk = tag[id];
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
        train(s, c, nc, dim, kc, num, tag);
        stop = calE(nc, c, dim, kc, e);
        copy(nc, c, dim, kc);
        it++;
    } while (!stop && it < maxIt);
}
int test() {
    hls::stream<ap_uint<sizeof(DType) * 8> > sampleStrm[PDV];
    hls::stream<bool> endSampleStrm;
    ap_uint<sizeof(DType) * 8> centers[PCU][PDV][UramDepth] = {0};
#pragma HLS array_partition variable = centers complete dim = 1
#pragma HLS array_partition variable = centers complete dim = 2
#pragma HLS bind_storage variable = centers impl = uram
    hls::stream<ap_uint<32> > tagStrm;
    hls::stream<bool> endTagStrm;

    int res = 0;
    int dim = D - 5;      // D - 5;
    int kcluster = K - 3; // K - 3;
    int nsample = N;
    const int sz = sizeof(DType) * 8;
    int maxIter = 10; // 1000;
    DType eps = 1e-8;
    ap_uint<32>* tag = (ap_uint<32>*)malloc(sizeof(ap_uint<32>) * N);

    ap_uint<32>* idx = (ap_uint<32>*)malloc(sizeof(ap_uint<32>) * N);
    ap_uint<32>* tg1 = (ap_uint<32>*)malloc(sizeof(ap_uint<32>) * N);
    ap_uint<32>* tg2 = (ap_uint<32>*)malloc(sizeof(ap_uint<32>) * N);
    int cdsz = D * sizeof(DType);
    DType** x = (DType**)malloc(sizeof(DType*) * N);
    for (int i = 0; i < N; i++) {
        x[i] = (DType*)malloc(cdsz);
    }
    DType** c = (DType**)malloc(sizeof(DType*) * K);
    for (int i = 0; i < K; i++) {
        c[i] = (DType*)malloc(cdsz);
    }
    DType** nc = (DType**)malloc(sizeof(DType*) * K);
    for (int i = 0; i < K; i++) {
        nc[i] = (DType*)malloc(cdsz);
    }
    DType** gnc = (DType**)malloc(sizeof(DType*) * K);
    for (int i = 0; i < K; i++) {
        gnc[i] = (DType*)malloc(cdsz);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            x[i][j] = 0.5 + (i * 131 + j) % 1000;
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < D; ++j) {
            DType t = 0.25 + (i * 131 + j) % 1000;
            c[i][j] = t;
        }
    }

    std::cout << "-------  computing golden ----" << std::endl;
    goldenTrain(x, c, nc, dim, kcluster, nsample, tag, eps, maxIter);
    goldenPredict(x, nc, dim, kcluster, nsample, tag);

    std::cout << "K=" << K << "   N=" << N << std::endl;

    for (int k = 0; k < K; ++k) {
        std::cout << "k=" << k << "   c=(";
        for (int j = 0; j < D; ++j) {
            DType cd = nc[k][j];
            std::cout << cd;
            if (j < D - 1) std::cout << ",";
        }
        std::cout << ")" << std::endl;
    }

    convert2Strm(x, sampleStrm, endSampleStrm, dim, nsample);
    convert2Array(nc, centers, dim, kcluster);

    kmeansPredict(sampleStrm, endSampleStrm, centers, dim, kcluster, tagStrm, endTagStrm);

    int count = 0;
    int jj = 0;
    while (!endTagStrm.read()) {
        ap_uint<32> tg = tagStrm.read();
        ap_uint<32> g = tag[count];
        res = tg == g ? res : res + 1;
        std::cout << "tg=" << tg << "  g=" << g << std::endl;
        if (tg != g) {
            std::cout << "res=" << res << "  count=" << count << std::endl;
            idx[jj] = count;
            tg1[jj] = tg;
            tg2[jj] = g;
            jj++;
        }
        count++;
    }
    bool hit = checkEqualDis(x, nc, dim, idx, tg1, tg2, res);
    res = hit ? 0 : res;
    res = count == nsample ? res : res + 1;

    std::cout << "cout=" << count << "  nsample=" << nsample << std::endl;
    std::cout << "res=" << res << std::endl;

    for (int i = 0; i < N; i++) free(x[i]);

    for (int i = 0; i < K; i++) {
        free(c[i]);
        free(nc[i]);
    }
    free(x);
    free(c);
    free(nc);
    free(tag);
    free(tg1);
    free(tg2);
    free(idx);
    return res;
}

int main(int argc, char** argv) {
    return test();
}
#endif

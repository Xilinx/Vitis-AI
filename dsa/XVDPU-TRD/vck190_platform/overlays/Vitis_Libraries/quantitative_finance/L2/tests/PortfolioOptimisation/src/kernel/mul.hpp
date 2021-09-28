#include "po_kernel.hpp"

// n*m * m*1 = n*1
template <int N, int M, int P>
void mv_mul(float X[N][M], float Y[P], float Z[P], int n, int m) {
mv_mul_loop1:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        Z[i] = 0;
    }

mv_mul_loop2:
    for (int j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
    mv_mul_loop3:
        for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            Z[i] += X[i][j] * Y[j];
        }
    }
}

// 1*n * n*1 = single value
template <int N, int M, int P>
void vv_mul(float X[N], float Y[M], float Z[P], int max) {
    Z[0] = 0;
vv_mul_loop1:
    for (int i = 0; i < max; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        Z[0] += X[i] * Y[i];
    }
}

// 1*n * n*m = 1*m
template <int P, int N, int M, int R>
void vm_mul(float X[P], float Y[N][M], float Z[R], int n, int m) {
vm_mul_loop1:
    for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        Z[i] = 0;
    }

vm_mul_loop2:
    for (int j = 0; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
    vm_mul_loop3:
        for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            Z[i] += X[j] * Y[i][j];
        }
    }
}

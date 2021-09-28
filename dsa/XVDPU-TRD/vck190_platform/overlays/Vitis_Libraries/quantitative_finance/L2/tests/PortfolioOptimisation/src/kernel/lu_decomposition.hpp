#define TINY_AMOUNT 1e-6

template <int N>
void swap_rows(float M[][N], int r1, int r2, int dim) {
swap_rows_loop1:
    for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        double tmp = M[r1][i];
        M[r1][i] = M[r2][i];
        M[r2][i] = tmp;
    }
}

// subtract (value * row c) from row
template <int N>
void sub_row_by(float M[][N], int row, int c, float value, int dim) {
sub_row_by_loop1:
    for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        if (i >= c) {
            M[row][i] -= (M[c][i] * value);
        }
    }
}

template <int N>
int eliminate_at(float L[][N], float U[][N], float P[][N], int row, int dim) {
    int ret = 1;

eliminate_at_loop1:
    for (int c = 0; c < row; c++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
        float value = U[row][c] / U[c][c];
        sub_row_by<N>(U, row, c, value, dim);
        /* remember the multiplier */
        L[row][c] = value;
    }
    return ret;
}

template <int N>
int permute_find_row_to_swap(float M[][N], int row, int dim) {
    int res = -1;
permute_find_row_to_swap_loop1:
    for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        if (M[i][row] > TINY_AMOUNT && row != i) {
            res = i;
            break;
        }
    }
    return res;
}

template <int N>
int permute(float U[][N], float P[][N], int dim) {
permute_loop1:
    for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
        //#pragma HLS pipeline II=1
        if (U[i][i] <= TINY_AMOUNT) {
            int ret = permute_find_row_to_swap<N>(U, i, dim);
            if (ret == -1) {
                return 0;
            }
            swap_rows<N>(U, i, ret, dim);
            swap_rows<N>(P, i, ret, dim);
        }
    }
    return 1;
}

template <int N>
int lu_decomposition(float M[][N], float L[][N], float U[][N], float P[][N], int dim) {
// initialise the arrays
lu_decomposition_loop1:
    for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
    lu_decomposition_loop2:
        for (int j = 0; j < dim; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            U[i][j] = M[i][j];
            if (i != j) {
                L[i][j] = 0;
                P[i][j] = 0;
            } else {
                L[i][j] = 1;
                P[i][j] = 1;
            }
        }
    }

    // permute if any of the leading diagonal are zero
    if (!permute<N>(U, P, dim)) {
        return 0;
    }

lu_decomposition_loop3:
    for (int i = 1; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
        if (!eliminate_at<N>(L, U, P, i, dim)) {
            return 0;
        }
    }
    return 1;
}

template <int N>
void l_solver(float L[][N], float c[], float Y[], int dim) {
    float tmp[N];
#pragma HLS array_partition variable = tmp cyclic factor = 4

l_solver_loop1:
    for (int R = 0; R < dim; R++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        tmp[R] = 0;
    }
    Y[0] = c[0];

l_solver_loop2:
    for (int R = 0; R < dim; R++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
        if (R > 0) {
        l_solver_loop3:
            for (int r = 0; r < R; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
                tmp[R] = tmp[R] + (L[R][r] * Y[r]);
            }
            Y[R] = c[R] - tmp[R];
        }
    }
}

template <int N>
void u_solver(float U[][N], float c[], float X[], int dim) {
    float tmp[N];
#pragma HLS array_partition variable = tmp cyclic factor = 4
u_solver_loop1:
    for (int row = 0; row < dim; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        tmp[row] = 1 / U[row][row];
    }

// make the leading diagonal all 1's
u_solver_loop2:
    for (int row = 0; row < dim; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        c[row] = c[row] * tmp[row];
    }

u_solver_loop2a:
    for (int row = 0; row < dim; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
    u_solver_loop3:
        for (int col = 0; col < dim; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            if (col >= row) {
                U[row][col] = U[row][col] * tmp[row];
            }
        }
    }

u_solver_loop4:
    for (int R = 0; R < dim; R++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        tmp[R] = 0;
    }
    X[dim - 1] = c[dim - 1];

u_solver_loop5:
    for (int j = dim - 1; j >= 0; j--) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
    u_solver_loop6:
        for (int i = 0; i < dim; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            if (i < j) {
                tmp[i] = tmp[i] + (U[i][j] * X[j]);
            }
        }
        X[j - 1] = c[j - 1] - tmp[j - 1];
    }
}

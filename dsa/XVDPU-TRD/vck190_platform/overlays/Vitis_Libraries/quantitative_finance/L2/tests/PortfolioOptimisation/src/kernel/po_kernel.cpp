#include <iostream>
#include "po_kernel.hpp"
#include "lu_decomposition.hpp"
#include "mul.hpp"
#include "math.h"
#include "xf_fintech/covariance.hpp"

#define BURSTBUFFERSIZE 256

template <int M, int N>
void calculate_returns(float prices[], float returns[][N], int num_assets, int num_prices) {
calculate_returns_loop1:
    for (int col = 1; col < num_prices; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 1000 max = 1000
    calculate_returns_loop2:
        for (int row = 0; row < num_assets; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 8
#pragma HLS pipeline II = 8
            returns[row][col - 1] = (prices[row + col * num_assets] / prices[row + ((col - 1) * num_assets)]) - 1;
        }
    }
}

template <int N>
void calculate_mean_returns(float returns[][N], float mean_returns[], int num_assets, int num_returns) {
init_loop:
    for (int row = 0; row < num_assets; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        mean_returns[row] = 0;
    }

calculate_mean_returns_loop1:
    for (int col = 0; col < num_returns; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 1000 max = 1000
    calculate_mean_returns_loop2:
        for (int row = 0; row < num_assets; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            mean_returns[row] += returns[row][col];
        }
    }
calculate_mean_returns_loop3:
    for (int row = 0; row < num_assets; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        mean_returns[row] /= (num_returns);
    }
}

template <int N>
void calculate_excess_returns(float returns[][N], float mean_returns[], int num_assets, int num_returns) {
calculate_excess_returns_loop1:
    for (int j = 0; j < num_returns; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1000 max = 1000
    calculate_excess_returns_loop2:
        for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            returns[i][j] = returns[i][j] - mean_returns[i];
        }
    }
}

template <int N>
void form_a_from_covar(float A[][N + 1], float covar[][N], int num_assets) {
form_a_from_covar_loop1:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
    form_a_from_covar_loop2:
        for (int j = 0; j < num_assets; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            A[i][j] = covar[i][j] * 2;
        }
        A[i][num_assets] = 1;
        A[num_assets][i] = 1;
    }
    A[num_assets][num_assets] = 0;
}

template <int N>
void form_b(float b[N], int num_assets) {
form_b_loop1:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        b[i] = 0;
    }
    b[num_assets] = 1;
}

template <int N, int M>
float portfolio_expected_return(float weights[N], float mean_returns[M], int num_assets) {
    float ret[1];
    vv_mul<N, M, 1>(weights, mean_returns, ret, num_assets);
    return ret[0];
}

template <int P, int N, int M>
float portfolio_variance(float weights[P], float covar[N][M], int num_assets) {
    float ret[1];
    float tmp[M];
#pragma HLS array_partition variable = tmp cyclic factor = 4

    vm_mul<P, N, M, M>(weights, covar, tmp, num_assets, num_assets);
    vv_mul<M, P, 1>(tmp, weights, ret, num_assets);
    return ret[0];
}

template <int N>
void calc_gmvp(float covar[][N], float mean_returns[], float* weights, int num_assets) {
// +1 for the Lagrange Multipliers
#ifdef __SYNTHESIS__
    float A[N + 1][N + 1];
    float L[N + 1][N + 1];
    float U[N + 1][N + 1];
    float P[N + 1][N + 1];
    float b[N + 1];
    float Y[N + 1];
    float w[N + 1];
#else

    float* b = (float*)malloc((N + 1) * sizeof(float));
    float* Y = (float*)malloc((N + 1) * sizeof(float));
    float* w = (float*)malloc((N + 1) * sizeof(float));

    std::unique_ptr<float[][N + 1]> A_c(new float[N + 1][N + 1]);
    float(*A)[N + 1] = A_c.get();

    std::unique_ptr<float[][N + 1]> L_c(new float[N + 1][N + 1]);
    float(*L)[N + 1] = L_c.get();

    std::unique_ptr<float[][N + 1]> U_c(new float[N + 1][N + 1]);
    float(*U)[N + 1] = U_c.get();

    std::unique_ptr<float[][N + 1]> P_c(new float[N + 1][N + 1]);
    float(*P)[N + 1] = P_c.get();

#endif

#pragma HLS array_partition variable = A cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = A cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = L cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = L cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = U cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = U cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = P cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = P cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = b cyclic factor = 4
#pragma HLS array_partition variable = Y cyclic factor = 4
#pragma HLS array_partition variable = w cyclic factor = 4

    // form A from the covariance matrix
    form_a_from_covar<N>(A, covar, num_assets);

    // LU decomposition of A
    lu_decomposition<N + 1>(A, L, U, P, num_assets + 1);

    // form b
    form_b<N + 1>(b, num_assets);

    // form Pb
    float Pb[N + 1];
#pragma HLS array_partition variable = Pb cyclic factor = 4
    mv_mul<N + 1, N + 1, N + 1>(P, b, Pb, num_assets + 1, num_assets + 1);

    // solve LY = Pb for Y
    l_solver<N + 1>(L, Pb, Y, num_assets + 1);

    // Solve Uw = Y for w
    u_solver<N + 1>(U, Y, w, num_assets + 1);

    // calculate portfolio expected return and variance
    float exp_ret = portfolio_expected_return<N + 1, N>(w, mean_returns, num_assets);
    float var = portfolio_variance<N + 1, N, N>(w, covar, num_assets);

// write the output
calc_gmvp_loop1:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        weights[i] = w[i];
    }
    weights[num_assets] = exp_ret;
    weights[num_assets + 1] = var;
}

template <int N>
void form_eff_a_from_covar(float A[][N + 2], float covar[][N], float mean_returns[], int num_assets) {
    int i;
    int j;
form_eff_a_from_covar_loop1:
    for (i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
    form_eff_a_from_covar_loop2:
        for (j = 0; j < num_assets; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
            A[i][j] = covar[i][j] * 2;
        }
        A[i][j++] = mean_returns[i];
        A[i][j] = 1;
    }

form_eff_a_from_covar_loop3:
    for (j = 0; j < num_assets; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        A[i][j] = mean_returns[j];
    }
    A[i][j++] = 0;
    A[i++][j] = 0;

form_eff_a_from_covar_loop4:
    for (j = 0; j < num_assets; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        A[i][j] = 1;
    }
    A[i][j++] = 0;
    A[i][j] = 0;
}

template <int N>
void form_eff_b(float b[N], float target_return, int num_assets) {
    int i;
form_eff_b_loop1:
    for (i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        b[i] = 0;
    }
    b[i++] = target_return;
    b[i] = 1;
}

template <int N>
void calc_eff_portfolio(float covar[][N], float mean_returns[N], float* weights, float target_return, int num_assets) {
#ifdef __SYNTHESIS__

    float A[N + 2][N + 2];
    float L[N + 2][N + 2];
    float U[N + 2][N + 2];
    float P[N + 2][N + 2];

    float b[N + 2];
    float Y[N + 2];
    float w[N + 2];
#else
    float* b = (float*)malloc((N + 2) * sizeof(float));
    float* Y = (float*)malloc((N + 2) * sizeof(float));
    float* w = (float*)malloc((N + 2) * sizeof(float));

    std::unique_ptr<float[][N + 2]> A_c(new float[N + 2][N + 2]);
    float(*A)[N + 2] = A_c.get();

    std::unique_ptr<float[][N + 2]> L_c(new float[N + 2][N + 2]);
    float(*L)[N + 2] = L_c.get();

    std::unique_ptr<float[][N + 2]> U_c(new float[N + 2][N + 2]);
    float(*U)[N + 2] = U_c.get();

    std::unique_ptr<float[][N + 2]> P_c(new float[N + 2][N + 2]);
    float(*P)[N + 2] = P_c.get();
#endif

#pragma HLS array_partition variable = A cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = A cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = L cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = L cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = U cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = U cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = P cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = P cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = b cyclic factor = 4
#pragma HLS array_partition variable = Y cyclic factor = 4
#pragma HLS array_partition variable = w cyclic factor = 4

    // form A from the covariance matrix
    form_eff_a_from_covar<N>(A, covar, mean_returns, num_assets);

    // LU decomposition of A
    lu_decomposition<N + 2>(A, L, U, P, num_assets + 2);

    // form b
    form_eff_b<N + 2>(b, target_return, num_assets);

    // form Pb
    float Pb[N + 2];
#pragma HLS array_partition variable = Pb cyclic factor = 4
    mv_mul<N + 2, N + 2, N + 2>(P, b, Pb, num_assets + 2, num_assets + 2);

    // solve LY = Pb for Y
    l_solver<N + 2>(L, Pb, Y, num_assets + 2);

    // Solve Uw = Y for w
    u_solver<N + 2>(U, Y, w, num_assets + 2);

    // calculate portfolio expected return and variance
    float exp_ret = portfolio_expected_return<N + 2, N>(w, mean_returns, num_assets);
    float var = portfolio_variance<N + 2, N, N>(w, covar, num_assets);

// write the output
calc_eff_portfolio_loop1:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        weights[i] = w[i];
    }
    weights[num_assets] = exp_ret;
    weights[num_assets + 1] = var;
}

template <int N>
void calc_tan_portfolio(float covar[][N],
                        float mean_returns[N],
                        float* tan_weights,
                        float* rf_weights,
                        float risk_free_rate,
                        float target_return,
                        int num_assets) {
#ifdef __SYNTHESIS__
    float L[N][N];
    float U[N][N];
    float P[N][N];
    float Y[N];
    float w[N];
    float excess_returns[N];
    float Pexcess_returns[N];
#else
    float* Y = (float*)malloc(N * sizeof(float));
    float* w = (float*)malloc(N * sizeof(float));
    float* excess_returns = (float*)malloc(N * sizeof(float));
    float* Pexcess_returns = (float*)malloc(N * sizeof(float));

    std::unique_ptr<float[][N]> L_c(new float[N][N]);
    float(*L)[N] = L_c.get();
    std::unique_ptr<float[][N]> U_c(new float[N][N]);
    float(*U)[N] = U_c.get();
    std::unique_ptr<float[][N]> P_c(new float[N][N]);
    float(*P)[N] = P_c.get();
#endif

#pragma HLS array_partition variable = L cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = L cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = U cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = U cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = P cyclic factor = 4 dim = 1
#pragma HLS array_partition variable = P cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = Y cyclic factor = 4
#pragma HLS array_partition variable = w cyclic factor = 4
#pragma HLS array_partition variable = excess_returns cyclic factor = 4
#pragma HLS array_partition variable = Pexcess_returns cyclic factor = 4

// form excess returns
calc_tan_portfolio_loop1:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        excess_returns[i] = mean_returns[i] - risk_free_rate;
    }

    // LU decomposition of covariance matrix
    lu_decomposition<N>(covar, L, U, P, num_assets);

    // form Pexcess_returns
    mv_mul<N, N, N>(P, excess_returns, Pexcess_returns, num_assets, num_assets);

    // solve covarY = Pexcess_returns for Y
    l_solver<N>(L, Pexcess_returns, Y, num_assets);

    // Solve Uw = Y for w
    u_solver<N>(U, Y, w, num_assets);

    // divide by sum of weights
    float sum = 0;
calc_tan_portfolio_loop2:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        sum += w[i];
    }
calc_tan_portfolio_loop3:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        w[i] = w[i] / sum;
    }

    // calculate portfolio expected return and variance
    float exp_ret = portfolio_expected_return<N, N>(w, mean_returns, num_assets);
    float var = portfolio_variance<N, N, N>(w, covar, num_assets);
    float sharpe = (exp_ret - risk_free_rate) / sqrtf(var);

// write the output
calc_tan_portfolio_loop4:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        tan_weights[i] = w[i];
    }
    tan_weights[num_assets] = exp_ret;
    tan_weights[num_assets + 1] = var;
    tan_weights[num_assets + 2] = sharpe;

    // risk free + tangency for risk free rate and given return
    // calculate tangency weight = (target return - Rf) / (tangent return - Rf)
    float tangency_weight = (target_return - risk_free_rate) / (exp_ret - risk_free_rate);
calc_tan_portfolio_loop5:
    for (int i = 0; i < num_assets; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 1
        rf_weights[i] = w[i] * tangency_weight;
    }
}

extern "C" {

// calculate the optimal weights Aw = b
// where A = 2S 1
//           1' 0
// where S is the covariance matrix and 1 is a unit vector and 1' its transpose
//
// w the weights = w0   an b = 0
//                 w1          0
//                 .           .
//                 .           .
//                 wn          0
//                 L           1
// L is the Lagrange Multiplier which we can ignore
//
// Use LU decomposition + Gaussian Elimination
// Aw = b
// LUw = Pb where P is the permutation matrix
// Let UPw = Y and therefore LY = Pb
// Solve LY = Pb for Y
// Solve Uw = Y for w
// Because of the form of b and L, Y = b so we can skip the first solver and we get Uw = b
void po_kernel(float* prices,
               int num_assets,
               int num_prices,
               float target_return,
               float risk_free_rate,
               float* gmvp_weights,
               float* eff_weights,
               float* tan_weights,
               float* rf_weights) {
#pragma HLS INTERFACE m_axi port = prices offset = slave bundle = gmem0 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = gmvp_weights offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = eff_weights offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = tan_weights offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = rf_weights offset = slave bundle = gmem4

#pragma HLS INTERFACE s_axilite port = prices bundle = control
#pragma HLS INTERFACE s_axilite port = gmvp_weights bundle = control
#pragma HLS INTERFACE s_axilite port = eff_weights bundle = control
#pragma HLS INTERFACE s_axilite port = tan_weights bundle = control
#pragma HLS INTERFACE s_axilite port = rf_weights bundle = control

#pragma HLS INTERFACE s_axilite port = num_assets bundle = control
#pragma HLS INTERFACE s_axilite port = num_prices bundle = control
#pragma HLS INTERFACE s_axilite port = target_return bundle = control
#pragma HLS INTERFACE s_axilite port = risk_free_rate bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

// read prices from shared memory into local memory
// sum the prices per asset
// populate the returns array

#ifdef __SYNTHESIS__
    float working[MAX_ASSETS][MAX_RETURNS];
    float covar[MAX_ASSETS][MAX_ASSETS];

    float mean_returns[MAX_ASSETS];
    float p[MAX_ASSETS * MAX_PRICES];
    float w_gmvp[MAX_ASSETS + 2];
    float w_eff[MAX_ASSETS + 2];
    float w_tan[MAX_ASSETS + 3];
    float w_rf[MAX_ASSETS];
#else
    float* mean_returns = (float*)malloc(MAX_ASSETS * sizeof(float));
    float* p = (float*)malloc(MAX_ASSETS * MAX_PRICES * sizeof(float));
    float* w_gmvp = (float*)malloc((MAX_ASSETS + 2) * sizeof(float));
    float* w_eff = (float*)malloc((MAX_ASSETS + 2) * sizeof(float));
    float* w_tan = (float*)malloc((MAX_ASSETS + 3) * sizeof(float));
    float* w_rf = (float*)malloc(MAX_ASSETS * sizeof(float));

    std::unique_ptr<float[][MAX_RETURNS]> working_c(new float[MAX_ASSETS][MAX_RETURNS]);
    float(*working)[MAX_RETURNS] = working_c.get();
    std::unique_ptr<float[][MAX_ASSETS]> covar_c(new float[MAX_ASSETS][MAX_ASSETS]);
    float(*covar)[MAX_ASSETS] = covar_c.get();
#endif

#pragma HLS ARRAY_PARTITION variable = working cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = working cyclic factor = 8 dim = 2
#pragma HLS ARRAY_PARTITION variable = covar cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = covar cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = p cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = mean_returns cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = w_gmvp cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = w_eff cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = w_tan cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = w_rf cyclic factor = 4

    // copy from DDR to local
    int num = num_prices * num_assets;
main_loop1:
    for (int i = 0; i < num; i += BURSTBUFFERSIZE) {
#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 1024
        int chunk_size = BURSTBUFFERSIZE;
        if ((i + BURSTBUFFERSIZE) > num) {
            chunk_size = num - i;
        }
    main_loop2:
        for (int j = 0; j < chunk_size; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 256
            p[i + j] = prices[i + j];
        }
    }

    // calculate returns and mean returns
    int num_returns = num_prices - 1;
    calculate_returns<MAX_ASSETS, MAX_RETURNS>(p, working, num_assets, num_prices);
    calculate_mean_returns<MAX_RETURNS>(working, mean_returns, num_assets, num_returns);

    // calculate the excess returns
    calculate_excess_returns<MAX_RETURNS>(working, mean_returns, num_assets, num_returns);

    // calculate the covariance matrix
    xf::fintech::covCoreMatrix<float, MAX_ASSETS, MAX_RETURNS, 8, 4>(num_assets, num_returns, working, covar);

    calc_gmvp<MAX_ASSETS>(covar, mean_returns, w_gmvp, num_assets);
    calc_eff_portfolio<MAX_ASSETS>(covar, mean_returns, w_eff, target_return, num_assets);
    calc_tan_portfolio(covar, mean_returns, w_tan, w_rf, risk_free_rate, target_return, num_assets);

// write the output
main_loop3:
    for (int i = 0; i < num_assets + 2; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS pipeline II = 1
        gmvp_weights[i] = w_gmvp[i];
        eff_weights[i] = w_eff[i];
        tan_weights[i] = w_tan[i];
        if (i < num_assets) {
            rf_weights[i] = w_rf[i];
        }
    }
    tan_weights[num_assets + 2] = w_tan[num_assets + 2];
}

} // extern C

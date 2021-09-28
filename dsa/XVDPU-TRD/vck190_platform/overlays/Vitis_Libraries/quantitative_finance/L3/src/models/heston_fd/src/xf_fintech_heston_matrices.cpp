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

#include <cmath>
#include <cstdlib>
#include <map>
#include <vector>

#include "xf_fintech_heston_coeffs.hpp"
#include "xf_fintech_heston_matrices.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

void Matrices::coeffsInit(void) {
    Coeffients.init();
}

void Matrices::insertOrUpdate(std::map<std::pair<int, int>, double>& sparse_map_A, int row, int col, double val) {
    std::pair<int, int> temp(row, col);
    sparse_map_A[temp] += val;
}

void Matrices::createA(std::map<std::pair<int, int>, double>& sparse_map_A,
                       std::vector<std::vector<double> >& vec_A1,
                       std::vector<std::vector<double> >& vec_A2) {
    /* Calculates the A0, A1, A2 matrices
     * Returns :
     * A - sparse matrix in S - major order
     * A1_vec - A1 diagonal vectors in S - major order
     * A2_vec - A2 diagonal vectors in v - major order
     */
    int i, j, k, l, row, col;
    int m1 = solverParams.m1;
    int m2 = solverParams.m2;
    double c, val, a, b;

    /* A0 contribution to A matrix - this is the mixed derivative term
     * Start both ranges at 1 as mixed term is zeroed by s = 0 and /or v = 0
     * End both terms at end - 1 as the mixed derivative is zero implied by
     * Neumann boundary condition(2.4)
     */
    for (j = 1; j < m2 - 1; j++) {
        for (i = 1; i < m1 - 1; i++) {
            c = modelParams.rho * modelParams.sig * sGrid.at(i) * vGrid.at(j);
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    row = i + j * m1;
                    col = (i + k) + (j + l) * m1;
                    val = c * Coeffients.sBeta(i, k) * Coeffients.vBeta(j, l);
                    insertOrUpdate(sparse_map_A, row, col, val);
                }
            }
        }
    }

    /* A1 contribution to A matrix plus separate tridiagonal representation -
     * terms in S
     * For s = S i.e.A[m1], boundary condition applies for du / ds
     * At s = 0, u is zero so don't need to worry about the rdU term here
     */
    for (j = 0; j < m2 - 1; j++) {
        for (i = 1; i < m1 - 1; i++) {
            a = 0.5 * pow(sGrid.at(i), 2) * vGrid.at(j);         // d2u / ds2 term
            b = (modelParams.rd - modelParams.rf) * sGrid.at(i); // du / ds term
            for (k = -1; k <= 1; k++) {
                row = i + j * m1;
                col = (i + k) + j * m1;
                val = a * Coeffients.sDelta(i, k) + b * Coeffients.sBeta(i, k);
                insertOrUpdate(sparse_map_A, row, col, val);
                vec_A1.at(row).at(k + 1) += val;
            }
            row = i + j * m1;
            col = i + j * m1;
            val = -0.5 * modelParams.rd;
            insertOrUpdate(sparse_map_A, row, col, val);
            vec_A1.at(row).at(1) += val;
        }

        /* Evaluate d2/ds2 term at Smax using virtualpoint (see virtualpoint.pdf)
         * This is i=m1 which was not calculated above
         * Use virtual point
         */
        row = (m1 - 1) + j * m1;
        a = 0.5 * pow(sGrid.at(m1 - 1), 2) * vGrid.at(j); // d2u / ds2 term

        col = (m1 - 2) + j * m1;
        val = (2 * a) / pow(Coeffients.sDx(m1 - 1), 2);
        insertOrUpdate(sparse_map_A, row, col, val);
        vec_A1.at(row).at(0) += val;

        col = (m1 - 1) + j * m1;
        val = (-2 * a) / pow(Coeffients.sDx(m1 - 1), 2);
        insertOrUpdate(sparse_map_A, row, col, val);
        vec_A1.at(row).at(1) += val;

        // also need the rdU term at s=S
        row = (m1 - 1) + j * m1;
        col = (m1 - 1) + j * m1;
        val = -0.5 * modelParams.rd;
        insertOrUpdate(sparse_map_A, row, col, val);
        vec_A1.at(row).at(1) += val;
    }

    /* A2 contribution to A matrix plus separate pentadiagonal representation -
     * terms in V
     * A2 vector is ordered differently with v inner so can't use row,col for that
     * assignment
     */
    double temp, temp2;

    for (j = 0; j < m2 - 1; j++) {
        for (i = 0; i < m1; i++) {
            temp = modelParams.kappa * (modelParams.eta - vGrid.at(j)); // First order term
            temp2 = 0.5 * pow(modelParams.sig, 2) * vGrid.at(j);        // Second order term
            if (vGrid.at(j) == 0) {
                //	 Scheme 2.9c
                // Only 1st derivative here as at v = 0, second derivative term
                // vanishes(v*d2u / dv2)
                for (k = 0; k <= 2; k++) {
                    row = i + m1 * j;
                    col = i + m1 * (k + j);
                    val = temp * Coeffients.vGamma(j, k);
                    insertOrUpdate(sparse_map_A, row, col, val);
                    vec_A2.at(j + i * m2).at(k + 2) += val;
                }
            } else if ((vGrid.at(j) > 1.0) && (vGrid.at(j) > modelParams.eta)) {
                // du / dv term  in v > 1 region when v > theta
                // Scheme 2.9a
                for (k = -2; k <= 0; k++) {
                    row = i + m1 * (j);
                    col = i + m1 * (k + j);
                    val = temp * Coeffients.vAlpha(j, k);
                    insertOrUpdate(sparse_map_A, row, col, val);
                    vec_A2.at(j + i * m2).at(k + 2) += val;
                }
                // d2u / dv2
                // Normal central limit
                for (k = -1; k <= 1; k++) {
                    row = i + m1 * (j);
                    col = i + m1 * (k + j);
                    val = temp2 * Coeffients.vDelta(j, k);
                    insertOrUpdate(sparse_map_A, row, col, val);
                    vec_A2.at(j + i * m2).at(k + 2) += val;
                }
            } else {
                // All other points, standard central limit for du / dv and d2u / dv2
                for (k = -1; k <= 1; k++) {
                    row = i + m1 * j;
                    col = i + m1 * (k + j);
                    val = temp * Coeffients.vBeta(j, k) + temp2 * Coeffients.vDelta(j, k);
                    insertOrUpdate(sparse_map_A, row, col, val);
                    vec_A2.at(j + i * m2).at(k + 2) += val;
                }
            }
            row = i + j * m1;
            col = i + j * m1;
            val = -0.5 * modelParams.rd;
            insertOrUpdate(sparse_map_A, row, col, val);
            vec_A2.at(j + i * m2).at(2) += val;
        }
    }

    return;
}

void Matrices::createB(std::vector<double>& vec_b) {
    int i, j;
    int m1 = solverParams.m1;
    int m2 = solverParams.m2;
    int m = m1 * m2;
    double val;

    for (j = 0; j < m2; j++) {
        val = (modelParams.rd - modelParams.rf) * sGrid.at(m1 - 1); // This is the constant part of the du/ds
                                                                    // expression.  The time dependent exp(-rf.t) is
                                                                    // done by F()
        val += 0.5 * pow(sGrid.at(m1 - 1), 2) * vGrid.at(j) *
               (2 / Coeffients.sDx(m1 - 1)); // Additional constant due to the virtual point extrapolation
        vec_b.at(j * m1 + m1 - 1) += val;
    }

    for (i = 0; i < m1; i++) {
        val = -0.5 * modelParams.rd * sGrid.at(i);
        vec_b.at(m - m1 - 2 + i) += val;
    }

    for (i = 0; i < m1; i++) {
        val = -0.5 * modelParams.rd * sGrid.at(i);
        vec_b.at(m - m1 - 2 + i) += val;
    }

    return;
}

} // namespace hestonfd
} // namespace fintech
} // namespace xf

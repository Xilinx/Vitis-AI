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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "xf_fintech_heston_adi_solver.hpp"
#include "xf_fintech_heston_kernel_interface.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

// DEBUG
//#define PRINT_CSV

void AdiSolver::createGrid(void) {
    if (AdiSolverParams.gridType == 0) {
        createUniformGrid();
    } else if (AdiSolverParams.gridType == 1) {
        createSinhGrid();
    }
}

void AdiSolver::solve(cl::Context* pContext, cl::CommandQueue* pCommandQueue, cl::Kernel* pKernel, double* u) {
    unsigned int i, j;
    unsigned int m1 = AdiSolverParams.m1;
    unsigned int m2 = AdiSolverParams.m2;
    unsigned int N = AdiSolverParams.N;
    unsigned int m = m1 * m2;
    double theta = AdiSolverParams.theta;
    double dt = AdiSolverParams.dt;

    std::map<std::pair<int, int>, double> sparse_map_A;
    std::vector<double> tempVec31(3);
    std::vector<std::vector<double> > A1_vec(m, tempVec31);
    std::vector<double> tempVec5(5);
    std::vector<std::vector<double> > A2_vec(m, tempVec5);

    std::vector<double> b(m);
    std::vector<double> u0(m);

    std::vector<double> tempVec32(3);
    std::vector<std::vector<double> > X1(m, tempVec32);
    std::vector<double> tempVec51(5);
    std::vector<std::vector<double> > X2(m, tempVec51);

    // Set up matrices and boundary conditions
    Matrices matrixGen(sGrid, vGrid, _sDelta, _vDelta, AdiModelParams, AdiSolverParams);
    matrixGen.coeffsInit();
    matrixGen.createA(sparse_map_A, A1_vec, A2_vec);
    matrixGen.createB(b);

    std::map<std::pair<int, int>, double>::iterator it;
    for (it = sparse_map_A.begin(); it != sparse_map_A.end(); it++) {
        it->second = it->second * dt;
    }

    for (i = 0; i < A1_vec.size(); i++) {
        for (j = 0; j < 3; j++) {
            X1.at(i).at(j) = 0.0 - theta * dt * A1_vec.at(i).at(j);
            if (j == 1) {
                X1.at(i).at(j) = X1.at(i).at(j) + 1;
            }
            A1_vec.at(i).at(j) = A1_vec.at(i).at(j) * dt * theta;
        }
    }

    for (i = 0; i < A2_vec.size(); i++) {
        for (j = 0; j < 5; j++) {
            X2.at(i).at(j) = 0.0 - theta * dt * A2_vec.at(i).at(j);
            if (j == 2) {
                X2.at(i).at(j) = X2.at(i).at(j) + 1;
            }
            A2_vec.at(i).at(j) = A2_vec.at(i).at(j) * dt * theta;
        }
    }

    for (i = 0; i < b.size(); i++) {
        b.at(i) = dt * b.at(i);
    }

    for (i = 0; i < m2; i++) {
        for (j = 0; j < m1; j++) {
            u0.at((i * m1) + j) = (sGrid.at(j) - AdiModelParams.K) > 0 ? (sGrid.at(j) - AdiModelParams.K) : 0;
        }
    }

#ifdef PRINT_CSV

    std::ofstream myfile;
    myfile.open("cplusplus_A.csv");
    myfile << sparse_map_A.size() << "\n";
    for (auto elem : sparse_map_A) {
        myfile << std::setprecision(20) << elem.first.first << "," << elem.first.second << "," << elem.second << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_A1.csv");
    for (i = 0; i < A1_vec.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << A1_vec[i][0] << "," << A1_vec[i][1] << "," << A1_vec[i][2]
               << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_A2.csv");
    for (i = 0; i < A2_vec.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << A2_vec[i][0] << "," << A2_vec[i][1] << "," << A2_vec[i][2]
               << "," << A2_vec[i][3] << "," << A2_vec[i][4] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_b.csv");
    for (i = 0; i < b.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << b[i] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_X1.csv");
    for (i = 0; i < X1.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << X1[i][0] << "," << X1[i][1] << "," << X1[i][2] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_X2.csv");
    for (i = 0; i < X2.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << X2[i][0] << "," << X2[i][1] << "," << X2[i][2] << ","
               << X2[i][3] << "," << X2[i][4] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_u0.csv");
    for (i = 0; i < u0.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << u0[i] << "\n";
    }
    myfile.close();
#endif

    if ((pContext != nullptr) && (pCommandQueue != nullptr) && (pKernel != nullptr)) {
        xf::fintech::hestonfd::kernel_call(pContext, pCommandQueue, pKernel, sparse_map_A, A1_vec, A2_vec, X1, X2, b,
                                           u0, m1, m2, N, u);
    } else {
        xf::fintech::hestonfd::kernel_call(sparse_map_A, A1_vec, A2_vec, X1, X2, b, u0, m1, m2, N, u);
    }
}

void AdiSolver::solve(double* u) {
    unsigned int i, j;
    unsigned int m1 = AdiSolverParams.m1;
    unsigned int m2 = AdiSolverParams.m2;
    unsigned int N = AdiSolverParams.N;
    unsigned int m = m1 * m2;
    double theta = AdiSolverParams.theta;
    double dt = AdiSolverParams.dt;

    std::map<std::pair<int, int>, double> sparse_map_A;
    std::vector<double> tempVec31(3);
    std::vector<std::vector<double> > A1_vec(m, tempVec31);
    std::vector<double> tempVec5(5);
    std::vector<std::vector<double> > A2_vec(m, tempVec5);

    std::vector<double> b(m);
    std::vector<double> u0(m);

    std::vector<double> tempVec32(3);
    std::vector<std::vector<double> > X1(m, tempVec32);
    std::vector<double> tempVec51(5);
    std::vector<std::vector<double> > X2(m, tempVec51);

    // Set up matrices and boundary conditions
    Matrices matrixGen(sGrid, vGrid, _sDelta, _vDelta, AdiModelParams, AdiSolverParams);
    matrixGen.coeffsInit();
    matrixGen.createA(sparse_map_A, A1_vec, A2_vec);
    matrixGen.createB(b);

    std::map<std::pair<int, int>, double>::iterator it;
    for (it = sparse_map_A.begin(); it != sparse_map_A.end(); it++) {
        it->second = it->second * dt;
    }

    for (i = 0; i < A1_vec.size(); i++) {
        for (j = 0; j < 3; j++) {
            X1.at(i).at(j) = 0.0 - theta * dt * A1_vec.at(i).at(j);
            if (j == 1) {
                X1.at(i).at(j) = X1.at(i).at(j) + 1;
            }
            A1_vec.at(i).at(j) = A1_vec.at(i).at(j) * dt * theta;
        }
    }

    for (i = 0; i < A2_vec.size(); i++) {
        for (j = 0; j < 5; j++) {
            X2.at(i).at(j) = 0.0 - theta * dt * A2_vec.at(i).at(j);
            if (j == 2) {
                X2.at(i).at(j) = X2.at(i).at(j) + 1;
            }
            A2_vec.at(i).at(j) = A2_vec.at(i).at(j) * dt * theta;
        }
    }

    for (i = 0; i < b.size(); i++) {
        b.at(i) = dt * b.at(i);
    }

    for (i = 0; i < m2; i++) {
        for (j = 0; j < m1; j++) {
            u0.at((i * m1) + j) = (sGrid.at(j) - AdiModelParams.K) > 0 ? (sGrid.at(j) - AdiModelParams.K) : 0;
        }
    }

#ifdef PRINT_CSV

    std::ofstream myfile;
    myfile.open("cplusplus_A.csv");
    myfile << sparse_map_A.size() << "\n";
    for (auto elem : sparse_map_A) {
        myfile << std::setprecision(20) << elem.first.first << "," << elem.first.second << "," << elem.second << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_A1.csv");
    for (i = 0; i < A1_vec.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << A1_vec[i][0] << "," << A1_vec[i][1] << "," << A1_vec[i][2]
               << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_A2.csv");
    for (i = 0; i < A2_vec.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << A2_vec[i][0] << "," << A2_vec[i][1] << "," << A2_vec[i][2]
               << "," << A2_vec[i][3] << "," << A2_vec[i][4] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_b.csv");
    for (i = 0; i < b.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << b[i] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_X1.csv");
    for (i = 0; i < X1.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << X1[i][0] << "," << X1[i][1] << "," << X1[i][2] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_X2.csv");
    for (i = 0; i < X2.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << X2[i][0] << "," << X2[i][1] << "," << X2[i][2] << ","
               << X2[i][3] << "," << X2[i][4] << "\n";
    }
    myfile.close();

    myfile.open("cplusplus_u0.csv");
    for (i = 0; i < u0.size(); i++) {
        myfile << std::scientific << std::setprecision(18) << u0[i] << "\n";
    }
    myfile.close();
#endif

    // Kernel call
    xf::fintech::hestonfd::kernel_call(sparse_map_A, A1_vec, A2_vec, X1, X2, b, u0, m1, m2, N, u);
}

double AdiSolver::createUniformGrid(void) {
    return 0;
}

// This function to be done once in software.
void AdiSolver::createSinhGrid(void) {
    int m1 = AdiSolverParams.m1;
    int m2 = AdiSolverParams.m2;
    double K = AdiModelParams.K;
    double S = AdiModelParams.K * 8;
    double V = AdiModelParams.V * 5;

    double c = K / 5;
    double d = V / 500;

    double dxi;

    std::vector<double> xi(m1);
    std::vector<double> sVec(m1);
    std::vector<double> sDelta(m1 - 1);
    std::vector<double> vVec(m2);
    std::vector<double> vDelta(m2 - 1);
    int i;

    dxi = (1.0 / (m1 - 1)) * (asinh((S - K) / c) - asinh(-K / c));

    for (i = 0; i < m1; i++) {
        xi.at(i) = asinh(-K / c) + i * dxi;
    }

    for (i = 0; i < m1; i++) {
        sVec.at(i) = K + (c * sinh((long double)xi.at(i)));
    }
    sVec.at(0) = 0;

    for (i = 0; i < (m1 - 1); i++) {
        sDelta.at(i) = sVec.at(i + 1) - sVec.at(i);
    }

    dxi = (1.0 / (m2 - 1)) * asinh(V / d);
    for (i = 0; i < m2; i++) {
        xi.at(i) = i * dxi;
    }

    for (i = 0; i < m2; i++) {
        vVec.at(i) = d * sinh((long double)xi.at(i));
    }

    vVec.at(0) = 0;

    for (i = 0; i < (m2 - 1); i++) {
        vDelta.at(i) = vVec.at(i + 1) - vVec.at(i);
    }

    sGrid = sVec;
    vGrid = vVec;
}

} // namespace hestonfd
} // namespace fintech
} // namespace xf

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

#ifndef _XF_FINTECH_HESTON_ADI_SOLVER_H_
#define _XF_FINTECH_HESTON_ADI_SOLVER_H_

#include "xf_fintech_heston.hpp"
#include "xf_fintech_heston_matrices.hpp"

#include "xcl2.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

class AdiSolver {
   private:
    std::vector<double> _sDelta;
    std::vector<double> _vDelta;
    model_parameters_t AdiModelParams;
    solver_parameters_t AdiSolverParams;

   public:
    std::vector<double> sGrid;
    std::vector<double> vGrid;

    AdiSolver(model_parameters_t modelParams, solver_parameters_t solverParams)
        : AdiModelParams(modelParams), AdiSolverParams(solverParams){};
    void createGrid(void);

    void solve(double* u);
    void solve(cl::Context* pContext, cl::CommandQueue* pCommandQueue, cl::Kernel* pKernel, double* u);

    double createUniformGrid();

    // This function to be done once in software.
    void createSinhGrid();
};

} // namespace hestonfd
} // namespace fintech
} // namespace xf

#endif //_XF_FINTECH_HESTON_ADI_SOLVER_H_

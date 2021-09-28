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
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/

#include "xf_fintech_heston.hpp"
#include <iostream>
#include "xf_fintech_heston_adi_solver.hpp"
#include "xf_fintech_heston_types.hpp"

using namespace xf::fintech::hestonfd;

namespace xf {
namespace fintech {

HestonFD::HestonFDReturnVal HestonFD::Solve(HestonFDPriceRam& AnyPriceRam,
                                            std::vector<double>& S,
                                            std::vector<double>& V) {
    HestonFDReturnVal Result = XLNXOK;

    _ExecutionTime.Start();

    double* results_u = AnyPriceRam.Get_PriceGrid();
    model_parameters_t modelParams;
    solver_parameters_t solverParams;

    if (HestonFD::_SolverParameters.Get_Scheme() == "Douglas") {
        solverParams.scheme = 1;
    }
    solverParams.theta = HestonFD::_SolverParameters.Get_Theta();
    solverParams.N = HestonFD::_SolverParameters.Get_N();   // Number of timesteps
    solverParams.dt = HestonFD::_SolverParameters.Get_dt(); // Delta(timestep) modelParams->T / solverParams.N
    solverParams.m1 = HestonFD::_SolverParameters.Get_m1(); // Number of grid steps in S direction
    solverParams.m2 = HestonFD::_SolverParameters.Get_m2(); // Number of grid steps in V direction
    solverParams.gridType = HestonFD::_SolverParameters.Get_GridType(); // 0 = uniform, 1 = sinh grid

    modelParams.K = HestonFD::_ModelParameters.Get_K();
    modelParams.kappa = HestonFD::_ModelParameters.Get_kappa();
    modelParams.rd = HestonFD::_ModelParameters.Get_rd();
    modelParams.rf = HestonFD::_ModelParameters.Get_rf();
    modelParams.rho = HestonFD::_ModelParameters.Get_rho();
    modelParams.S = HestonFD::_ModelParameters.Get_S();
    modelParams.sig = HestonFD::_ModelParameters.Get_sig();
    modelParams.T = HestonFD::_ModelParameters.Get_T();
    modelParams.eta = HestonFD::_ModelParameters.Get_eta();
    modelParams.V = HestonFD::_ModelParameters.Get_V();

    AdiSolver solver(modelParams, solverParams);
    solver.createGrid();
    solver.solve(_OCLObjects.GetContext(), _OCLObjects.GetCommandQueue(), _OCLObjects.GetKernel(), results_u);

    S = solver.sGrid;
    V = solver.vGrid;

    _Solved = true;

    _ExecutionTime.Stop();

    return Result;
}

HestonFD::HestonFDReturnVal HestonFD::Meta(std::chrono::milliseconds& AnyExecutionTime) {
    HestonFDReturnVal Result = XLNXOK;

    if (_Solved) {
        AnyExecutionTime = _ExecutionTime.Duration();
    } else {
        AnyExecutionTime = (std::chrono::milliseconds)0;
        Result = XLNXAlgorithmNotExecuted;
    }

    return Result;
}

} // namespace fintech
} // namespace xf

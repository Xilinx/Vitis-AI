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

#ifndef _XLNX_SOLVER_PARAMETERS_
#define _XLNX_SOLVER_PARAMETERS_

#include <string>

#include "xf_fintech_heston_model_parameters.hpp"

using namespace std;

namespace xf {
namespace fintech {

/**
 * @brief Heston FD Solver Parameters Class
 */

class HestonFDSolverParameters {
   public:
    /** @brief XLNXHestonSolverParameters class constructor */
    HestonFDSolverParameters(HestonFDModelParameters& ModelParameters);

    /** @brief get the theta */
    double Get_Theta() { return _Theta; }
    /** @brief set the theta */
    void Set_Theta(double theta) { _Theta = theta; }

    /** @brief get the number of timesteps */
    int Get_N() { return _N; }
    /** @brief set the number of timesteps */
    void Set_N(int N) { _N = N; }

    /** @brief get the delta timestep */
    double Get_dt() { return _dt; }
    /** @brief set the delta timestep */
    void Set_dt(int dt) { _dt = dt; }

    /** @brief get the m1 grid size for the S direction */
    int Get_m1() { return _m1; }
    /** @brief set the m1 grid size for the S direction */
    void Set_m1(int m1) { _m1 = m1; }

    /** @brief get the m2 grid size for the V direction */
    int Get_m2() { return _m2; }
    /** @brief set the m2 grid size for the V direction */
    void Set_m2(int m2) { _m2 = m2; }

    /** @brief GridType enum */
    enum GridType { uniform = 0, sinh };

    string Get_Scheme() { return _DouglasScheme; }
    GridType Get_GridType() { return _Default_GridType; }

   private:
    const string _DouglasScheme = "Douglas";
    const GridType _Default_GridType = GridType::sinh;

    const double _Default_Theta = 0.5;
    const int _Default_N = 200;
    const int _Default_m1 = 128;
    const int _Default_m2 = 64;

    double _Theta;
    int _N;
    double _dt;
    int _m1;
    int _m2;
};

} // namespace fintech
} // namespace xf

#endif // _XLNX_SOLVER_PARAMETERS_

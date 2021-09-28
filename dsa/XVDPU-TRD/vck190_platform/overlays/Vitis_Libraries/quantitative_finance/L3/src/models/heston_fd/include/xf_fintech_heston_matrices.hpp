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

#ifndef _XF_FINTECH_HESTON_MATRICES_H_
#define _XF_FINTECH_HESTON_MATRICES_H_

#include <map>
#include <vector>

#include "xf_fintech_heston_coeffs.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

class Matrices {
   private:
    Coeffs Coeffients;
    std::vector<double> sGrid;
    std::vector<double> vGrid;
    model_parameters_t modelParams;
    solver_parameters_t solverParams;

    void insertOrUpdate(std::map<std::pair<int, int>, double>& sparse_map_A, int row, int col, double val);

   public:
    Matrices(std::vector<double> sGrid,
             std::vector<double> vGrid,
             std::vector<double> sDelta,
             std::vector<double> vDelta,
             model_parameters_t modelParams,
             solver_parameters_t solverParams)
        : Coeffients(sGrid, vGrid, sDelta, vDelta),
          sGrid(sGrid),
          vGrid(vGrid),
          modelParams(modelParams),
          solverParams(solverParams) {}

    void coeffsInit(void);
    void createA(std::map<std::pair<int, int>, double>& sparse_map_A,
                 std::vector<std::vector<double> >& vec_A1,
                 std::vector<std::vector<double> >& vec_A2);
    void createB(std::vector<double>& vec_b);
};

} // namespace hestonfd
} // namespace fintech
} // namespace xf

#endif //_XF_FINTECH_HESTON_MATRICES_H_

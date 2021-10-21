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

#ifndef _XF_FINTECH_HESTON_HPP_
#define _XF_FINTECH_HESTON_HPP_

#include <chrono>
#include <vector>

#include "xf_fintech_heston_execution_time.hpp"
#include "xf_fintech_heston_model_parameters.hpp"
#include "xf_fintech_heston_ocl_objects.hpp"
#include "xf_fintech_heston_price_ram.hpp"
#include "xf_fintech_heston_solver_parameters.hpp"

using namespace std;

namespace xf {
namespace fintech {

/** @brief Heston class.

    Heston FD main interface class.
*/

class HestonFD {
   public:
    enum HestonFDReturnVal { XLNXOK = 0, XLNXAlgorithmNotExecuted };

    /** @brief Class constructor instantiated with an instance of
     * XLNXHestonModelParameters & XLNXHestonSolverParameters
     */
    HestonFD(HestonFDModelParameters& AnyModelParameters, HestonFDSolverParameters& AnySolverParameters)
        : _ModelParameters(AnyModelParameters), _SolverParameters(AnySolverParameters){};

    HestonFD(HestonFDModelParameters& AnyModelParameters,
             HestonFDSolverParameters& AnySolverParameters,
             HestonFDOCLObjects& AnyOCLObjects)
        : _ModelParameters(AnyModelParameters), _SolverParameters(AnySolverParameters), _OCLObjects(AnyOCLObjects){};

    /** @brief Solve Heston FD, returns the full price grid, vector of the stock
     * price and vector of variance values
     */
    HestonFDReturnVal Solve(HestonFDPriceRam& AnyPriceRam, std::vector<double>& S, std::vector<double>& V);

    /** @brief Return the time taken to solve Heston FD
     */
    HestonFDReturnVal Meta(std::chrono::milliseconds& AnyExecutionTime);

   private:
    HestonFDModelParameters& _ModelParameters;
    HestonFDSolverParameters& _SolverParameters;
    HestonFDOCLObjects _OCLObjects;
    HestonFDExecutionTime _ExecutionTime;
    bool _Solved = false;
};

} // namespace fintech
} // namespace xf

#endif // _XF_FINTECH_HESTON_

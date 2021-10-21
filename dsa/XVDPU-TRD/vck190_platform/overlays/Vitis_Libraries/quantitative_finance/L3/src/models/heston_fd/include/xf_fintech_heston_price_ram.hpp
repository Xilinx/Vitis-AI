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

#ifndef _XF_FINTECH_HESTON_PRICE_RAM_
#define _XF_FINTECH_HESTON_PRICE_RAM_

#include "xf_fintech_heston_solver_parameters.hpp"

namespace xf {
namespace fintech {

using namespace std;

/**
 * @brief Heston FD Model Price Ram Class.
 */

class HestonFDPriceRam {
   public:
    /** @brief Class Constructor for memory allocation */
    HestonFDPriceRam(HestonFDSolverParameters& AnySolverParameters) : _SolverParameters(AnySolverParameters) {
        CreatePriceRAM();
    };

    /** @brief Class Destructor to free memory */
    ~HestonFDPriceRam();

    /** @brief Check if memory has been successfully aquired */
    bool RAM_Is_Acquired() { return _RAM_Acquired; }

    /** @brief return m1 grid size */
    int Get_m1_GridSize() { return _StockPrice_m1_GridSize; }

    /** @brief return m2 grid size */
    int Get_m2_GridSize() { return _StockPrice_m2_GridSize; }

    /** @brief return pointer to the price grid */
    double* Get_PriceGrid() { return _PriceGrid; }

   private:
    bool _RAM_Acquired = false;

    int _StockPrice_m1_GridSize;
    int _StockPrice_m2_GridSize;
    double* _PriceGrid = nullptr;

    HestonFDSolverParameters& _SolverParameters;

    int Calculate_m2_Grid_Size();
    int Calculate_m1_Grid_Size();
    void CreatePriceRAM();
};

} // namespace fintech
} // namespace xf

#endif // _XF_FINTECH_HESTON_PRICE_RAM_

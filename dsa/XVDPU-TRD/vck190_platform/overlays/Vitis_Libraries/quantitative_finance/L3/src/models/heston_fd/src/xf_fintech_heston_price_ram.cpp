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

#include "xf_fintech_heston_price_ram.hpp"

namespace xf {
namespace fintech {

HestonFDPriceRam::~HestonFDPriceRam() {
    delete[] _PriceGrid;
}

int HestonFDPriceRam::Calculate_m2_Grid_Size() {
    return (_SolverParameters.Get_m2());
}

int HestonFDPriceRam::Calculate_m1_Grid_Size() {
    return (_SolverParameters.Get_m1());
}

void HestonFDPriceRam::CreatePriceRAM() {
    _RAM_Acquired = true;
    _StockPrice_m1_GridSize = Calculate_m1_Grid_Size();
    _StockPrice_m2_GridSize = Calculate_m2_Grid_Size();

    if ((0 == _StockPrice_m1_GridSize) || (0 == _StockPrice_m2_GridSize)) {
        _RAM_Acquired = false;
    }

    if (_RAM_Acquired) {
        try {
            _PriceGrid = new double[_StockPrice_m1_GridSize * _StockPrice_m2_GridSize]();
        } catch (...) {
            _RAM_Acquired = false;
        }
    }
}

} // namespace fintech
} // namespace xf

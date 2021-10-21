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

#ifndef _XF_FINTECH_HESTON_COEFFS_H_
#define _XF_FINTECH_HESTON_COEFFS_H_

#include <vector>

#include "xf_fintech_heston_types.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

class Coeffs {
   private:
    std::vector<double> _sGrid;
    std::vector<double> _vGrid;
    std::vector<double> _sDelta;
    std::vector<double> _vDelta;

   public:
    Coeffs(std::vector<double> sGrid, std::vector<double> vGrid, std::vector<double> sDelta, std::vector<double> vDelta)
        : _sGrid(sGrid), _vGrid(vGrid), _sDelta(sDelta), _vDelta(vDelta) {}

    void init(void);

    double sDx(int pos);
    double sAlpha(int i, int pos);
    double vAlpha(int i, int pos);
    double sBeta(int i, int pos);
    double vBeta(int i, int pos);
    double sGamma(int i, int pos);
    double vGamma(int i, int pos);
    double sDelta(int i, int pos);
    double vDelta(int i, int pos);
    double alpha(std::vector<double> dx, int i, int pos);
    double beta(std::vector<double> dx, int i, int pos);
    double gamma(std::vector<double> dx, int i, int pos);
    double delta(std::vector<double> dx, int i, int pos);
};

} // namespace hestonfd
} // namespace fintech
} // namespace xf

#endif //_XF_FINTECH_HESTON_COEFFS_H_

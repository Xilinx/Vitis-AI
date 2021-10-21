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

#include <cstdlib>

#include "xf_fintech_heston_coeffs.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

void Coeffs::init(void) {
    std::vector<double> zeroArray(1);
    std::vector<double> diffArray;
    Diff(_sGrid, &diffArray);
    Concatenate(zeroArray, diffArray, &_sDelta);

    std::vector<double> vdiffArray;
    Diff(_vGrid, &vdiffArray);
    Concatenate(zeroArray, vdiffArray, &_vDelta);
}

double Coeffs::sDx(int pos) {
    return _sDelta[pos];
}

double Coeffs::sAlpha(int i, int pos) {
    return alpha(_sDelta, i, pos);
}

double Coeffs::vAlpha(int i, int pos) {
    return alpha(_vDelta, i, pos);
}

double Coeffs::sBeta(int i, int pos) {
    return beta(_sDelta, i, pos);
}

double Coeffs::vBeta(int i, int pos) {
    return beta(_vDelta, i, pos);
}

double Coeffs::sGamma(int i, int pos) {
    return gamma(_sDelta, i, pos);
}

double Coeffs::vGamma(int i, int pos) {
    return gamma(_vDelta, i, pos);
}

double Coeffs::sDelta(int i, int pos) {
    return delta(_sDelta, i, pos);
}

double Coeffs::vDelta(int i, int pos) {
    return delta(_vDelta, i, pos);
}

double Coeffs::alpha(std::vector<double> dx, int i, int pos) {
    double coeff = 0;

    if (pos == -2) {
        coeff = dx.at(i) / (dx.at(i - 1) * (dx.at(i - 1) + dx.at(i)));
    } else if (pos == -1) {
        coeff = (-dx.at(i - 1) - dx.at(i)) / (dx.at(i - 1) * dx.at(i));
    } else if (pos == 0) {
        coeff = (dx.at(i - 1) + 2 * dx.at(i)) / (dx.at(i) * (dx.at(i - 1) + dx.at(i)));
    }

    return coeff;
}

double Coeffs::beta(std::vector<double> dx, int i, int pos) {
    double coeff = 0;

    if (pos == -1) {
        coeff = -dx.at(i + 1) / (dx.at(i) * (dx.at(i) + dx.at(i + 1)));
    } else if (pos == 0) {
        coeff = (dx.at(i + 1) - dx.at(i)) / (dx.at(i) * dx.at(i + 1));
    } else if (pos == 1) {
        coeff = dx.at(i) / (dx.at(i + 1) * (dx.at(i) + dx.at(i + 1)));
    }

    return coeff;
}

double Coeffs::gamma(std::vector<double> dx, int i, int pos) {
    double coeff = 0;

    if (pos == 0) {
        coeff = (-2 * dx.at(i + 1) - dx.at(i + 2)) / (dx.at(i + 1) * (dx.at(i + 1) + dx.at(i + 2)));
    } else if (pos == 1) {
        coeff = (dx.at(i + 1) + dx.at(i + 2)) / (dx.at(i + 1) * dx.at(i + 2));
    } else if (pos == 2) {
        coeff = -dx.at(i + 1) / (dx.at(i + 2) * (dx.at(i + 1) + dx.at(i + 2)));
    }

    return coeff;
}

double Coeffs::delta(std::vector<double> dx, int i, int pos) {
    double coeff = 0;

    if (pos == -1) {
        coeff = 2 / (dx.at(i) * (dx.at(i) + dx.at(i + 1)));
    } else if (pos == 0) {
        coeff = -2 / (dx.at(i) * dx.at(i + 1));
    } else if (pos == 1) {
        coeff = 2 / (dx.at(i + 1) * (dx.at(i) + dx.at(i + 1)));
    }

    return coeff;
}

} // namespace hestonfd
} // namespace fintech
} // namespace xf

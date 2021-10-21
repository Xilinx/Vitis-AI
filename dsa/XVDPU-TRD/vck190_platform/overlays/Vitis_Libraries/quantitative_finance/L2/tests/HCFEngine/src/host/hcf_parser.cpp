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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "hcf.hpp"
#include "xcl2.hpp"

bool validate_parameters(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p) {
    if (p->K == -999999) {
        std::cout << "K" << std::endl;
        return false;
    }
    if (p->s0 == -999999) {
        std::cout << "s0" << std::endl;
        return false;
    }
    if (p->v0 == -999999) {
        std::cout << "v0" << std::endl;
        return false;
    }
    if (p->rho == -999999) {
        std::cout << "rho" << std::endl;
        return false;
    }
    if (p->vvol == -999999) {
        std::cout << "vvol" << std::endl;
        return false;
    }
    if (p->vbar == -999999) {
        std::cout << "vbar" << std::endl;
        return false;
    }
    if (p->T == -999999) {
        std::cout << "T" << std::endl;
        return false;
    }
    if (p->r == -999999) {
        std::cout << "r" << std::endl;
        return false;
    }
    if (p->kappa == -999999) {
        std::cout << "kappa" << std::endl;
        return false;
    }
    if (p->dw == -999999) {
        std::cout << "dw" << std::endl;
        return false;
    }
    if (p->w_max == -999999) {
        std::cout << "w_max" << std::endl;
        return false;
    }
    return true;
}

void init_parameters(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p, TEST_DT dw, int w_max) {
    p->K = -999999;
    p->s0 = -999999;
    p->v0 = -999999;
    p->rho = -999999;
    p->vvol = -999999;
    p->vbar = -999999;
    p->T = -999999;
    p->r = -999999;
    p->kappa = -999999;
    p->dw = dw;
    p->w_max = w_max;
}

bool process_token(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p,
                   std::string token,
                   int line_num,
                   TEST_DT* expected_value) {
    std::istringstream iss_token(token);
    std::string parm;
    if (!std::getline(iss_token, parm, '=')) {
        std::cout << "ERROR(" << line_num << "): incorrectly formed token: " << token << std::endl;
        return false;
    }

    std::string value;
    if (!std::getline(iss_token, value, '=')) {
        std::cout << "ERROR(" << line_num << "): incorrectly formed token: " << token << std::endl;
        return false;
    }

    std::string dummy;
    if (std::getline(iss_token, dummy, '=')) {
        std::cout << "ERROR(" << line_num << "): incorrectly formed token: " << token << std::endl;
        return false;
    }

    try {
        if (parm.compare("K") == 0) {
            p->K = std::stod(value);
        } else if (parm.compare("s0") == 0) {
            p->s0 = std::stod(value);
        } else if (parm.compare("v0") == 0) {
            p->v0 = std::stod(value);
        } else if (parm.compare("rho") == 0) {
            p->rho = std::stod(value);
        } else if (parm.compare("vvol") == 0) {
            p->vvol = std::stod(value);
        } else if (parm.compare("vbar") == 0) {
            p->vbar = std::stod(value);
        } else if (parm.compare("T") == 0) {
            p->T = std::stod(value);
        } else if (parm.compare("r") == 0) {
            p->r = std::stod(value);
        } else if (parm.compare("kappa") == 0) {
            p->kappa = std::stod(value);
        } else if (parm.compare("exp") == 0) {
            *expected_value = std::stod(value);
        } else {
            std::cout << "ERROR: unknown parm: " << parm << std::endl;
            ;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: not a number: " << value << " : exception: " << e.what() << std::endl;
        return false;
    }

    return true;
}

bool is_comment(std::string line) {
    bool ret = false;
    std::size_t found = line.find_first_not_of(" \n\r\t");
    if (line[found] == '#') {
        ret = true;
    }
    return ret;
}

bool process_line(std::string line,
                  struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p,
                  TEST_DT* expected_value,
                  int line_num) {
    std::istringstream iss_line(line);
    std::string token;
    while (std::getline(iss_line, token, ' ')) {
        if (!process_token(p, token, line_num, expected_value)) {
            return false;
        }
    }
    if (!validate_parameters(p)) {
        std::cout << "ERROR: invalid parameters:" << std::endl;
        return false;
    }
    return true;
}

bool parse_file(std::string file,
                std::vector<struct xf::fintech::hcfEngineInputDataType<TEST_DT>,
                            aligned_allocator<struct xf::fintech::hcfEngineInputDataType<TEST_DT> > >& p,
                TEST_DT dw,
                int w_max,
                int* num_tests,
                TEST_DT* expected_values,
                int max_entries) {
    std::ifstream ifs(file, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << "ERROR: Failed to open file:" << file << std::endl;
        return false;
    }

    std::string line;
    int n = 0;
    while (std::getline(ifs, line)) {
        if (!is_comment(line)) {
            if (n >= max_entries) {
                *num_tests = max_entries + 1;
                return false;
            }
            init_parameters(&p[n], dw, w_max);
            if (process_line(line, &p[n], &expected_values[n], n + 1) == false) {
                return false;
            }
            n++;
        }
    }
    *num_tests = n;
    return true;
}

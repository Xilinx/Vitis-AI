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
#ifndef SVD_FUNCTION_H
#define SVD_FUNCTION_H

#include "util.hpp"
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

//! store kernel execution results to a csv file
void generate_output_file(int run, double* output_data, int output_size, std::string operation, std::string file_path);

void benchmark_svd_functions(std::string xclbinName, double& errA);

#endif

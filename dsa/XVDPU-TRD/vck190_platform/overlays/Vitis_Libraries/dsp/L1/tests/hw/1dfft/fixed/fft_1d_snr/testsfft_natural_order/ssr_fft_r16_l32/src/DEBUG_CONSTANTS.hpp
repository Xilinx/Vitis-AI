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
//================================== End Lic =================================================
// File Name : DEBUG_CONSTANTS.hpp
#ifndef DEBUG_CONSTANTS_H_
#define DEBUG_CONSTANTS_H_

/*********************************************************************
 * NO_DATA_FRAMES_TO_SIMULATE : is number of times
 * FFT is simulated.
 *********************************************************************/
#define NO_DATA_FRAMES_TO_SIMULATE 5

/*********************************************************************
 * MAX_PERCENT_ERROR_IN_SAMPLE is the maximum allowed error in percent
 * when comparing with golden output if sample differ by less than
 * MAX_PERCENT_ERROR_IN_SAMPLE it is not counted as error
 *********************************************************************/
#define MAX_PERCENT_ERROR_IN_SAMPLE 10

/*********************************************************************
 * MAX_ALLOWED_PERCENTAGE_OF_SAMPLES_IN_ERROR is %tage of total errors
 * allowed in any simulation(per frame) to pass if the errors in any
 * frame (number of mismatches) are larger than
 * MAX_ALLOWED_PERCENTAGE_OF_SAMPLES_IN_ERROR, the simulation fails.
 * ******************************************************************/
#define MAX_ALLOWED_PERCENTAGE_OF_SAMPLES_IN_ERROR 5

#define CHECK_COVEARAGE                                                                        \
    std::cout << "\n\n\n\nCovered;;;;;;;;\n" << __FILE__ << "Line:" << __LINE__ << "<<\n\n\n"; \
    exit(1)

#endif // DEBUG_CONSTANTS_H

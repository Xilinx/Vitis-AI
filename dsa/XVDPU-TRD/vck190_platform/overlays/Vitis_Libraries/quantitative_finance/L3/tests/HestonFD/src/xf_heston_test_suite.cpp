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

#include "xf_fintech_heston_test_suite.hpp"

int HestonFDTestSuite::Run(CSV csv, double delta) {
    int i;
    int ret = 0; // assume pass

    try {
        std::cout << csv.showNumberOfTests() << " Testcases Found:" << std::endl;

        std::cout << "################################################################" << std::endl;
        for (i = 0; i < csv.showNumberOfTests(); i++) {
            std::cout << "Running testcase " << csv.showTestCase(i) << std::endl; // results
            HestonFDTestCase* MyTestCase = new HestonFDTestCase(delta, m_xclbin, m_device);
            int mismatches = MyTestCase->Run(csv.showTestCase(i), csv.showTestScheme(i), csv.showTestParameters(i));
            csv.setNumberMismatches(i, mismatches);
            delete MyTestCase;
            if (mismatches) {
                ret = 1;
            }
        }

        // Summary of testcases
        std::cout << "################################################################" << std::endl;
        std::cout << "Summary Testcase Results" << std::endl;
        std::cout << "Delta:" << delta << std::endl;
        std::cout << "################################################################" << std::endl;
        for (i = 0; i < csv.showNumberOfTests(); i++) {
            std::cout << "TestCase: " << csv.showTestCase(i)
                      << " Number Mismatches Found:" << csv.getNumberMismatches(i) << std::endl; // results
        }
        std::cout << "################################################################" << std::endl << std::endl;

    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
    } catch (const std::string& ex) {
        std::cout << ex << std::endl;
    } catch (...) {
        std::cout << "Exception Found" << std::endl;
    }

    return ret;
}

HestonFDTestSuite::~HestonFDTestSuite() {}


/**
 * Copyright 2019 Xilinx Inc.
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

#include "apm.hpp"

#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <thread>

#ifdef BOARD_ULTRA96
#define APM_CLK_FREQ 266666656
#else
#define APM_CLK_FREQ 533333333
#endif

int main(int argc, char *argv[])
{
    APM apm(1);

    // Set metrics for test
    apm.set_metrics_counter(1, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_0);
    apm.set_metrics_counter(1, XAPM_METRIC_WRITE_BEAT_COUNT, XAPM_METRIC_COUNTER_1);
    apm.set_metrics_counter(2, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_2);
    apm.set_metrics_counter(2, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_3);
    apm.set_metrics_counter(3, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_4);
    apm.set_metrics_counter(3, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_5);
    apm.set_metrics_counter(4, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_6);
    apm.set_metrics_counter(4, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_7);
    apm.set_metrics_counter(5, XAPM_METRIC_READ_BYTE_COUNT, XAPM_METRIC_COUNTER_8);
    apm.set_metrics_counter(5, XAPM_METRIC_WRITE_BYTE_COUNT, XAPM_METRIC_COUNTER_9);
    
    auto t1 = std::chrono::steady_clock::now().time_since_epoch();
    double t2 =  std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1>>>(t1).count();
    printf("%.7f\n", t2);
    //auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_to);

    apm.start_collect(APM_CLK_FREQ);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    apm.stop_collect();

    return EXIT_SUCCESS;
}

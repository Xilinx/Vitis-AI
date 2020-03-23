/*
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

#ifdef BOARD_ULTRA96
#define APM_CLK_FREQ 266666656
#else
#define APM_CLK_FREQ 533333333
#endif
extern "C" {
    APM apm(1);

    int apm_start(double interval_in_sec)
    {
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
        
	unsigned int interval_clk = (unsigned int)(APM_CLK_FREQ * interval_in_sec);
	//std::cout << interval_clk << std::endl;
        apm.start_collect(interval_clk);
    
        return EXIT_SUCCESS;
    }

    int apm_stop()
    {
        apm.stop_collect();

	return EXIT_SUCCESS;
    }

    int apm_pop_data(struct apm_record *d)
    {
   	return apm.pop_data(d); 
    }
}

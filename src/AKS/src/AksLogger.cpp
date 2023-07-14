/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include "aks/AksLogger.h"

namespace AKS {

LogLevel minloglevel = LogLevel::INFO;

void Logger::setMinLogLevel() {
    minloglevel = LogLevel::INFO;
    char* llstr = std::getenv("AKS_VERBOSE");
    if (!llstr) {
        // _WARNING << "Environment variable AKS_VERBOSE not set. Setting
        // default to level 1 (INFO)" << std::endl;
        return;
    }

    int ll = std::atoi(llstr);
    ll = ll > 5 ? 5 : ll;
    minloglevel = static_cast<LogLevel>(ll);
    LOG_X(DEBUG) << "LogLevel : " << static_cast<int>(minloglevel) << std::endl;
}

}

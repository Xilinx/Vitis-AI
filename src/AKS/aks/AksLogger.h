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
#ifndef __AKS_LOGGER_H_
#define __AKS_LOGGER_H_

#include <string>
#include <iostream>
using namespace std;

namespace AKS {

enum class LogLevel {
    FULL = 2,
    DEBUG = 2,
    INFO = 1,
    WARNING = 0,
    ERROR = 0,
    FATAL = 0
};

extern LogLevel minloglevel;

#define _DEBUG std::cout << "[DEBUG] "
#define _INFO std::cout << "[INFO] "
#define _WARNING std::cout << "\033[1;95m[WARNING]\033[0m "
#define _ERROR std::cout << "\033[1;91m[ERROR]\033[0m "
#define _FATAL std::cout << "\033[1;91m[FATAL]\033[0m "

// Use this for single log commands
// Eg : LOG(INFO) << "Hello\n";
#define LOG_X(x) \
    if (AKS::LogLevel::x <= AKS::minloglevel) _##x
// Use this for bulk log commands
// This avoids multiple condition checks for perf sake
// Eg : SLOG(LogLevel::INFO,
//          _INFO << "Hello\n";
//          _INFO << "Worldn"; )
#define SLOG(level, cmd)             \
    if (level <= AKS::minloglevel) { \
        cmd                          \
    }

class Logger {
   public:
    static void setMinLogLevel();
};
}  // namespace AKS

#define INIT_LOGGING() AKS::LogLevel AKS::minloglevel = AKS::LogLevel::INFO;

#define SET_LOGGING_LEVEL() AKS::Logger::setMinLogLevel();

#endif

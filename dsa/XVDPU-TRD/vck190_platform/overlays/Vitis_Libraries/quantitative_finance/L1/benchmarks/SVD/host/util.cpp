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
#include "util.hpp"
#include <iostream>

unsigned long diff(const struct timeval* newTime, const struct timeval* oldTime) {
    return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}

int read_verify_env_int(const char* var, int fail_value) {
    if (getenv(var) == NULL) {
        std::cerr << "Warning, environment variable " << var << " not set, using " << fail_value << std::endl;
        return fail_value;
    } else
        return atoi(getenv(var));
}

std::string read_verify_env_string(const char* var, std::string fail_value) {
    if (getenv(var) == NULL) {
        return fail_value;
    } else
        return getenv(var);
}

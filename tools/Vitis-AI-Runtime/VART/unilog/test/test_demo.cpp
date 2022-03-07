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

#include <iostream>
#include <memory>
#include <string>
using namespace std;

#include "UniLog/UniLog.hpp"

int main(int argc, char *argv[]) {
  UniLog::Initial(
      argv[0],
      // UNI_LOG_STD, // toggle for only std output, default
      // UNI_LOG_FILE,              // toggle for only file output
      UNI_LOG_STD_FILE, // toggle for both stderr and file
      // UNI_LOG_CHECK_FAKE_LAYER, // toggle for check the random fake layer
      // UNI_LOG_LEVEL_INFO,   // set only catch the log above info
      // UNI_LOG_LEVEL_WARNING, // set only catch the log above warning
      // UNI_LOG_LEVEL_ERROR,       // set only catch the log above error
      // UNI_LOG_LEVEL_FATAL,       // set only catch the log above fatal
      UNI_LOG_STD_LEVEL_INFO, // set only display the log above info
      // UNI_LOG_STD_LEVEL_WARNING, // set only display the log above warning
      // UNI_LOG_STD_LEVEL_ERROR,   // set only display the log above error
      // UNI_LOG_STD_LEVEL_FATAL,   // set only display the log above fatal
      UNI_LOG_LOW, // set to display verbosity above low
      // UNI_LOG_HIGH,        // set to display verbosity above high
      UNI_LOG_DEFAULT_PATH // set the log path to default, which is "./log"
  );
  // UniLog::setUniLogPath("./logself/");
  UniLog::setUniLogPrefix("SELFDEFINE");
  UNI_LOG_INFO << "This is a UniLog INFO";
  UNI_LOG_WARNING << "This is a UniLog WARNING";
  UNI_LOG_ERROR(ERROR_SAMPLE) << "This is a UniLog ERROR";
  // UNI_LOG_FATAL("Core Dump") << "This is a UniLog FATAL";
  UNI_LOG_VINFO(UNI_LOG_LOW) << "This is a verbosity low info!";
  UNI_LOG_VINFO(UNI_LOG_HIGH) << "This is a verbosity high info!";
  UNI_LOG_INFO_LOW << "This is a verbosity low info, use syntax sugar!";
  UNI_LOG_INFO_HIGH << "This is a verbosity high info, use syntax sugar!";

  UNI_LOG_DEBUG_INFO << "This is a debug INFO";
  UNI_LOG_DEBUG_WARNING << "This is a debug WARNING";
  UNI_LOG_DEBUG_ERROR(ERROR_SAMPLE) << "This is a debug ERROR";
  // UNI_LOG_DEBUG_FATAL("Core Dump") << "This is a debug FATAL";
  // UNI_LOG_DEBUG_CHECK(1 > 3) << "left must be larger than right";
  UNI_LOG_CHECK(1 > 2, ERROR_SAMPLE) << "Must be larger";
  unique_ptr<int> uptr;
  UNI_LOG_CHECK(uptr, ERROR_SAMPLE) << "This pointer is empty";
  return 0;
}

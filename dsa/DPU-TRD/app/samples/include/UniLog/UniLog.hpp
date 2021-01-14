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

#pragma once

// UNI_LOG_DEBUG_MODE : toggle of debug mode
#if defined(UNI_LOG_NDEBUG)
#undef UNI_LOG_DEBUG_MODE
#else
#define UNI_LOG_DEBUG_MODE
#endif

#include "ErrorCode.hpp"
#include <glog/logging.h>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>
using namespace std;

// APIs for release versions
#define UNI_LOG_INFO LOG(INFO) << "[" << UniLog::UniLogPrefix << "][INFO] "
#define UNI_LOG_WARNING                                                        \
  LOG(WARNING) << "[" << UniLog::UniLogPrefix << "][WARNING] "
#ifndef UNI_LOG_DEBUG_MODE
#define UNI_LOG_ERROR(ERRID)                                                   \
  LOG(ERROR) << "[" << UniLog::UniLogPrefix << "][ERROR]"                      \
             << "[" << GEN_ERROR(ERRID).getErrID() << "]"                      \
             << "[" << GEN_ERROR(ERRID).getErrDsp() << "] "
#define UNI_LOG_FATAL(ERRID)                                                   \
  LOG(FATAL) << "[" << UniLog::UniLogPrefix << "][FATAL]"                      \
             << "[" << GEN_ERROR(ERRID).getErrID() << "]"                      \
             << "[" << GEN_ERROR(ERRID).getErrDsp() << "] "
#else
#define UNI_LOG_ERROR(ERRID)                                                   \
  LOG(ERROR) << "[" << UniLog::UniLogPrefix << "][ERROR][" << __FILE__ << ":"  \
             << __LINE__ << "]"                                                \
             << "[" << GEN_ERROR(ERRID).getErrID() << "]"                      \
             << "[" << GEN_ERROR(ERRID).getErrDsp() << "]"                     \
             << "[" << GEN_ERROR(ERRID).getErrDebugInfo() << "] "
#define UNI_LOG_FATAL(ERRID)                                                   \
  LOG(FATAL) << "[" << UniLog::UniLogPrefix << "][FATAL][" << __FILE__ << ":"  \
             << __LINE__ << "]"                                                \
             << "[" << GEN_ERROR(ERRID).getErrID() << "]"                      \
             << "[" << GEN_ERROR(ERRID).getErrDsp() << "]"                     \
             << "[" << GEN_ERROR(ERRID).getErrDebugInfo() << "] "
#endif

// offered to use the verbosity info
#define UNI_LOG_VINFO(VERBOSITY)                                               \
  LOG_IF(INFO, ((int)VERBOSITY - (int)UNI_LOG_LOW > UniLog::UniLogVerbosity))  \
      << "[" << UniLog::UniLogPrefix << "][INFO] "
// syntax sugar for verbosity info
#define UNI_LOG_INFO_LOW UNI_LOG_VINFO(UNI_LOG_LOW)
#define UNI_LOG_INFO_HIGH UNI_LOG_VINFO(UNI_LOG_HIGH)

// UNI_LOG_CHECK is used for always check, it gives the no-confidential info,
// but give the developer the debug details
#define UNI_LOG_CHECK(condition, ERRID)                                        \
  UniLog::mapLogConfig[UNI_LOG_CHECK_FAKE_LAYER]                               \
      ? UNI_LOG_DEBUG_CHECK_RP(condition, ERRID)                               \
      : UNI_LOG_IF(condition, ERRID)

// UNI_LOG_DEBUG_CHECK only valide in debug mode, but eliminated in ndebug mode
#if defined(UNI_LOG_DEBUG_MODE)
#define UNI_LOG_DEBUG_CHECK(condition, ERRID) UNI_LOG_IF(condition, ERRID)
#else
#define UNI_LOG_DEBUG_CHECK(condition, ERRID)                                  \
  true ? (void)0 : google::LogMessageVoidify() & LOG(INFO)
#endif
// UNI_LOG_DEBUG_CHECK_RP is used to throw an exception for generate the fake
// layer
#define UNI_LOG_DEBUG_CHECK_RP(condition, ERRID)                               \
  check_throw(!!(condition), GEN_ERROR(ERRID))
#define UNI_LOG_CHECK_THROW(condition, ERRID)                                  \
  Checker(!!(condition)) & GEN_ERROR(ERRID)

#if defined(UNI_LOG_DEBUG_MODE)
#define UNI_LOG_IF(condition, ERRID)                                           \
  LOG_IF(FATAL, GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!(condition)))                 \
      << "[" << UniLog::UniLogPrefix << "][DEBUG-FATAL]"                       \
      << "[" << __FILE__ << ":" << __LINE__ << "]"                             \
      << "[Check Failed: " #condition "]"                                      \
      << "[" << GEN_ERROR(ERRID).getErrID() << "]"                             \
      << "[" << GEN_ERROR(ERRID).getErrDsp() << "]"                            \
      << "[" << GEN_ERROR(ERRID).getErrDebugInfo() << "] "
#else
#define UNI_LOG_IF(condition, ERRID)                                           \
  LOG_IF(FATAL, GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!(condition)))                 \
      << "[" << UniLog::UniLogPrefix << "][FATAL]"                             \
      << "[" << GEN_ERROR(ERRID).getErrID() << "]"                             \
      << "[" << GEN_ERROR(ERRID).getErrDsp() << "] "
#endif

// APIs only for debug mode
#if defined(UNI_LOG_DEBUG_MODE)
#define UNI_LOG_DEBUG_LOG_HANDLE(severity) LOG(severity)
#else
#define UNI_LOG_DEBUG_LOG_HANDLE(severity)                                     \
  true ? (void)0 : google::LogMessageVoidify() & LOG(severity)
#endif
#define UNI_LOG_DEBUG_INFO                                                     \
  UNI_LOG_DEBUG_LOG_HANDLE(INFO)                                               \
      << "[" << UniLog::UniLogPrefix << "][DEBUG-INFO]"                        \
      << "[" << __FILE__ << ":" << __LINE__ << "] "
#define UNI_LOG_DEBUG_WARNING                                                  \
  UNI_LOG_DEBUG_LOG_HANDLE(WARNING)                                            \
      << "[" << UniLog::UniLogPrefix << "][DEBUG-WARNING]"                     \
      << "[" << __FILE__ << ":" << __LINE__ << "] "
#define UNI_LOG_DEBUG_ERROR(ERRID)                                             \
  UNI_LOG_DEBUG_LOG_HANDLE(ERROR)                                              \
      << "[" << UniLog::UniLogPrefix << "][DEBUG-ERROR]"                       \
      << "[" << __FILE__ << ":" << __LINE__ << "]"                             \
      << "[" << GEN_ERROR(ERRID).getErrID() << "]"                             \
      << "[" << GEN_ERROR(ERRID).getErrDsp() << "]"                            \
      << "[" << GEN_ERROR(ERRID).getErrDebugInfo() << "] "
#define UNI_LOG_DEBUG_FATAL(ERRID)                                             \
  UNI_LOG_DEBUG_LOG_HANDLE(FATAL)                                              \
      << "[" << UniLog::UniLogPrefix << "][DEBUG-FATAL]"                       \
      << "[" << __FILE__ << ":" << __LINE__ << "]"                             \
      << "[" << GEN_ERROR(ERRID).getErrID() << "]"                             \
      << "[" << GEN_ERROR(ERRID).getErrDsp() << "]"                            \
      << "[" << GEN_ERROR(ERRID).getErrDebugInfo() << "] "

// the enumerator for Log status set
enum UniLogSet {
  UNI_LOG_SET_MIN = 0,
  UNI_LOG_STD,               // toggle for only std output, default
  UNI_LOG_FILE,              // toggle for only file output
  UNI_LOG_STD_FILE,          // toggle for both stderr and file
  UNI_LOG_CHECK_FAKE_LAYER,  // toggle for check the random fake layer
  UNI_LOG_LEVEL_INFO,        // set only catch the log above info
  UNI_LOG_LEVEL_WARNING,     // set only catch the log above warning
  UNI_LOG_LEVEL_ERROR,       // set only catch the log above error
  UNI_LOG_LEVEL_FATAL,       // set only catch the log above fatal
  UNI_LOG_STD_LEVEL_INFO,    // set only display the log above info
  UNI_LOG_STD_LEVEL_WARNING, // set only display the log above warning
  UNI_LOG_STD_LEVEL_ERROR,   // set only display the log above error
  UNI_LOG_STD_LEVEL_FATAL,   // set only display the log above fatal
  UNI_LOG_LOW,  // set to display verbosity above low(exclude), default is null
  UNI_LOG_HIGH, // set to display verbosity above high(exlude)
  UNI_LOG_DEFAULT_PATH, // set the log path to default, which is "./log"
  UNI_LOG_SET_MAX
};

// handler class for the logging system
class UniLog {
public:
  template <class... ParaClass>
  static UniLog &Initial(char *argv, ParaClass... LogSet) {
    static UniLog logger(argv, {LogSet...});
    return logger;
  }
  static void setUniLogPath(const string &logpath);
  static void setUniLogPrefix(const string &prefix);
  static map<UniLogSet, bool> mapLogConfig;
  static string UniLogPath;
  static string UniLogPrefix;
  static int UniLogVerbosity;

private:
  UniLog(char *argv, vector<UniLogSet> LogSet);
  static bool checkConfig(const UniLogSet &cfg_field);
  void printDebugModeAlert();
  ~UniLog();
};

// This is the function to override the default failure handle function
void UniFailureSignalFunction();

// used to check condition and throw exception when failed
void check_throw(const bool &condition, ErrorCode &errCode);

// checker is used to help throw error.
class Checker {
public:
  Checker() = delete;
  Checker(const bool &condition);
  void operator&(ErrorCode &errCode);

private:
  bool condition_;
};

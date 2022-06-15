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

#include "UniLog/UniLog.hpp"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using std::map;
using std::string;
using std::vector;
map<UniLogSet, bool> UniLog::mapLogConfig = {};
string UniLog::UniLogPath = (fs::current_path() / "log").string();
string UniLog::UniLogPrefix = "UNILOG";
int UniLog::UniLogVerbosity = -1;

UniLog::UniLog(char *argv, vector<UniLogSet> LogSet) {
  // init the glog
  google::InitGoogleLogging(argv);
#ifndef UNI_LOG_DEBUG_MODE
  google::InstallFailureFunction(UniFailureSignalFunction);
#endif
  for (auto &ConfigFeild : LogSet) {
    mapLogConfig[ConfigFeild] = true;
  }
  // turn off the default log prefix of glog
  FLAGS_log_prefix = false;
  FLAGS_colorlogtostderr = true;

  if ((!checkConfig(UNI_LOG_FILE)) //
      && (!checkConfig(UNI_LOG_STD_FILE))) {
    // default, only to stderr
    FLAGS_logtostderr = true;
  } else if ((!checkConfig(UNI_LOG_STD))    //
             && (checkConfig(UNI_LOG_FILE)) //
             && (!checkConfig(UNI_LOG_STD_FILE))) {
    // only to file
    // create the directory in the current path named "log"
    // redirect the log file to the current path
    setUniLogPath(UniLogPath);
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = false;
  } else if ((!checkConfig(UNI_LOG_STD))     //
             && (!checkConfig(UNI_LOG_FILE)) //
             && (checkConfig(UNI_LOG_STD_FILE))) {
    // to both stderr and file
    setUniLogPath(UniLogPath);
    // FLAGS_logtostderr = true;
    // FLAGS_alsologtostderr = true;
  } else {
    std::cerr
        << "\033[31m"
        << "Log information direction configure error! Please only use one of "
           "the UNI_LOG_STD, UNI_LOG_FILE or UNI_LOG_STD_FILE!"
        << "\033[0m" << std::endl;
    abort();
  }

  // config the log verbosity
  if (checkConfig(UNI_LOG_LEVEL_INFO)) {
    FLAGS_minloglevel = google::GLOG_INFO;
  }
  if (checkConfig(UNI_LOG_LEVEL_WARNING)) {
    FLAGS_minloglevel = google::GLOG_WARNING;
  }
  if (checkConfig(UNI_LOG_LEVEL_ERROR)) {
    FLAGS_minloglevel = google::GLOG_ERROR;
  }
  if (checkConfig(UNI_LOG_LEVEL_FATAL)) {
    FLAGS_minloglevel = google::GLOG_FATAL;
  }

  if (checkConfig(UNI_LOG_STD_FILE)) {
    // only suitable under the stderr_file mode
    if (checkConfig(UNI_LOG_STD_LEVEL_INFO)) {
      FLAGS_stderrthreshold = google::GLOG_INFO;
    }
    if (checkConfig(UNI_LOG_STD_LEVEL_WARNING)) {
      FLAGS_stderrthreshold = google::GLOG_WARNING;
    }
    if (checkConfig(UNI_LOG_STD_LEVEL_ERROR)) {
      FLAGS_stderrthreshold = google::GLOG_ERROR;
    }
    if (checkConfig(UNI_LOG_STD_LEVEL_FATAL)) {
      FLAGS_stderrthreshold = google::GLOG_FATAL;
    }
  }

  // set the info verbosity
  if (checkConfig(UNI_LOG_LOW)) {
    UniLogVerbosity = (int)UNI_LOG_LOW - (int)UNI_LOG_LOW;
  }
  if (checkConfig(UNI_LOG_HIGH)) {
    UniLogVerbosity = (int)UNI_LOG_HIGH - (int)UNI_LOG_LOW;
  }
#ifdef UNI_LOG_DEBUG_MODE
  printDebugModeAlert();
#endif
}

bool UniLog::checkConfig(const UniLogSet &cfg_field) {
  return (mapLogConfig.count(cfg_field) && mapLogConfig[cfg_field]);
}

void UniLog::setUniLogPath(const string &logpath) {
  if (FLAGS_log_dir == logpath) {
    return;
  }
  UniLogPath = logpath;
  if (checkConfig(UNI_LOG_DEFAULT_PATH)) {
    if (fs::exists(UniLogPath)) {
      if (!fs::remove_all(UniLogPath)) {
        std::cerr << UniLogPath << " is not empty, and failed to remove it."
                  << std::endl;
        std::abort();
      }
    }
    if (!fs::create_directories(UniLogPath)) {
      std::cerr << "Failed to create the log directory at "
                << fs::absolute(UniLogPath) << "." << std::endl;
      std::abort();
    }
  }
  FLAGS_log_dir = UniLogPath;
}

void UniLog::setUniLogPrefix(const string &prefix) { UniLogPrefix = prefix; }

void UniLog::printDebugModeAlert() {
  // remind the programmer take care of the debug mode
  std::cout << "\033[31m"
            << "/*PROJECT IS UNDER DEBUG MODE*****************/" << std::endl;
  std::cout << "/*KEEP ALL THE FILE INTERNAL******************/" << std::endl;
  std::cout << "/*ONLY RELEASE EXECUTABLE UNDER NON-DEBUG MODE/"
            << "\033[0m" << std::endl;
  return;
}

UniLog::~UniLog() {
#ifdef UNI_LOG_DEBUG_MODE
  printDebugModeAlert();
#endif
  google::ShutdownGoogleLogging();
}

void UniFailureSignalFunction() {
  std::cerr << "This program has crashed!" << std::endl;
  abort();
}

void check_throw(const bool &condition, ErrorCode &errCode) {
  if (condition) {
    return;
  } else {
    throw errCode;
  }
}

Checker::Checker(const bool &condition) { this->condition_ = condition; }

void Checker::operator&(ErrorCode &errCode) {
  if (condition_) {
    return;
  } else {
    throw errCode;
  }
}

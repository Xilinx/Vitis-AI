/**
 * Copyright (C) 2018 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

// ------ I N C L U D E   F I L E S -------------------------------------------
// Local - Include Files
#include "XclBinUtilMain.h"

#include "FormattedOutput.h"
#include "ParameterSectionData.h"
#include "XclBinClass.h"
#include "XclBinSignature.h"
#include "XclBinUtilities.h"

// 3rd Party Library - Include Files
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <stdexcept>

// System - Include Files
#include <iostream>
#include <set>
#include <string>

namespace XUtil = XclBinUtilities;

namespace {
enum ReturnCodes {
  RC_SUCCESS = 0,
  RC_ERROR_IN_COMMAND_LINE = 1,
  RC_ERROR_UNHANDLED_EXCEPTION = 2,
};
}

void drcCheckFiles(const std::vector<std::string>& _inputFiles,
                   const std::vector<std::string>& _outputFiles, bool _bForce) {
  std::set<std::string> normalizedInputFiles;

  for (auto file : _inputFiles) {
    if (!boost::filesystem::exists(file)) {
      std::string errMsg =
          "ERROR: The following input file does not exist: " + file;
      throw std::runtime_error(errMsg);
    }
    boost::filesystem::path filePath(file);
    normalizedInputFiles.insert(canonical(filePath).string());
  }

  std::vector<std::string> normalizedOutputFiles;
  for (auto file : _outputFiles) {
    if (boost::filesystem::exists(file)) {
      if (_bForce == false) {
        std::string errMsg =
            "ERROR: The following output file already exists on disk (use the "
            "force option to overwrite): " +
            file;
        throw std::runtime_error(errMsg);
      } else {
        boost::filesystem::path filePath(file);
        normalizedOutputFiles.push_back(canonical(filePath).string());
      }
    }
  }

  // See if the output file will stomp on an input file
  for (auto file : normalizedOutputFiles) {
    if (normalizedInputFiles.find(file) != normalizedInputFiles.end()) {
      std::string errMsg =
          "ERROR: The following output file is also used for input : " + file;
      throw std::runtime_error(errMsg);
    }
  }
}

static bool bQuiet = false;
void QUIET(const std::string _msg) {
  if (bQuiet == false) {
    std::cout << _msg.c_str() << std::endl;
  }
}

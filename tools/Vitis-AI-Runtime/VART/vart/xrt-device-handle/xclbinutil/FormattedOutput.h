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

#ifndef __FormattedOutput_h_
#define __FormattedOutput_h_

// ----------------------- I N C L U D E S -----------------------------------
#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include "xclbin.h"


// #includes here - please keep these to a bare minimum!

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...
class Section;

// ------------------- C L A S S :   S e c t i o n ---------------------------

namespace FormattedOutput {
  void reportInfo(std::ostream &_ostream, const std::string& _sInputFile, const axlf &_xclBinHeader, const std::vector<Section*> _sections, bool _bVerbose);
  void getKernelDDRMemory(const std::string _sKernelInstanceName, const std::vector<Section*> _sections, boost::property_tree::ptree &_ptKernelInstance, boost::property_tree::ptree &_ptMemoryConnections);
  void reportVersion(bool _bShort = false);

  std::string getTimeStampAsString(const axlf &_xclBinHeader);
  std::string getFeatureRomTimeStampAsString(const axlf &_xclBinHeader);
  std::string getVersionAsString(const axlf &_xclBinHeader);
  std::string getMagicAsString(const axlf &_xclBinHeader);
  std::string getSignatureLengthAsString(const axlf &_xclBinHeader);
  std::string getKeyBlockAsString(const axlf &_xclBinHeader);
  std::string getUniqueIdAsString(const axlf &_xclBinHeader);
  std::string getModeAsString(const axlf &_xclBinHeader);
  std::string getFeatureRomUuidAsString(const axlf &_xclBinHeader);
  std::string getPlatformVbnvAsString(const axlf &_xclBinHeader);
  std::string getXclBinUuidAsString(const axlf &_xclBinHeader);
  std::string getDebugBinAsString(const axlf &_xclBinHeader);
}

#endif

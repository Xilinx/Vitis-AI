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
#include "SectionHeader.h"

#include <stdexcept>


SectionHeader::SectionHeader() 
  : m_eType(BITSTREAM)
{
  // Empty
}

SectionHeader::~SectionHeader() {
  // Empty
}

void
SectionHeader::readXclBinBinarySection(std::fstream& _istream, unsigned int _section) {
  // Find the section header data
  long long sectionOffset = sizeof(axlf) + (_section * sizeof(axlf_section_header)) - sizeof(axlf_section_header);
  _istream.seekg(sectionOffset);

  // Read in the data
  axlf_section_header sectionHeader = axlf_section_header {0};
  const unsigned int expectBufferSize = sizeof(axlf_section_header);

  _istream.read((char*)&sectionHeader, sizeof(axlf_section_header));

  if (_istream.gcount() != expectBufferSize) {
    std::string errMsg = "ERROR: Input stream is smaller then the expected section header size.";
    throw std::runtime_error(errMsg);
  }
}


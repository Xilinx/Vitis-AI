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

#include "SectionBitstream.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

// Static Variables / Classes
SectionBitstream::_init SectionBitstream::_initializer;

SectionBitstream::SectionBitstream() {
  // Empty
}

SectionBitstream::~SectionBitstream() {
  // Empty
}

std::string
SectionBitstream::getContentTypeAsString()
{
  if (m_bufferSize < 8) {
     return "Binary Image";
  }

  XUtil::TRACE_BUF("BUFFER", (const char*) m_pBuffer, 8);

  // Bitstream
  if (((unsigned char) m_pBuffer[0] == 0x00 ) &&
      ((unsigned char) m_pBuffer[1] == 0x09 ) &&
      ((unsigned char) m_pBuffer[2] == 0x0f ) &&
      ((unsigned char) m_pBuffer[3] == 0xf0 ) &&
      ((unsigned char) m_pBuffer[4] == 0x0f ) &&
      ((unsigned char) m_pBuffer[5] == 0xf0 ) &&
      ((unsigned char) m_pBuffer[6] == 0x0f ) &&
      ((unsigned char) m_pBuffer[7] == 0xf0 )) {
      return "Bitstream";
  }

  // ZIP
  if (((unsigned char) m_pBuffer[0] == 0x50 ) &&
      ((unsigned char) m_pBuffer[1] == 0x4B )) {
    if ((((unsigned char) m_pBuffer[2] == 0x03 ) &&
         ((unsigned char) m_pBuffer[3] == 0x04 )) ||
        (((unsigned char) m_pBuffer[2] == 0x05 ) &&
         ((unsigned char) m_pBuffer[3] == 0x06 )) ||
        (((unsigned char) m_pBuffer[2] == 0x07 ) &&
         ((unsigned char) m_pBuffer[3] == 0x08 ))) {
      if (m_name == "behav") {
        return "HW Emulation Binary";
      }
      return "DCP";
    }
  }

  // ELF
  if (((unsigned char) m_pBuffer[0] == 0x7f ) &&
      ((unsigned char) m_pBuffer[1] == 0x45 ) &&
      ((unsigned char) m_pBuffer[2] == 0x4c ) &&
      ((unsigned char) m_pBuffer[3] == 0x46 )) {
    return "SW Emulation Binary";
  }

  return "Binary Image";
}




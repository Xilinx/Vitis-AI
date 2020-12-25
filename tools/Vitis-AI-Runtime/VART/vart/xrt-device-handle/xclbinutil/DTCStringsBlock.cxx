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

#include "DTCStringsBlock.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;


DTCStringsBlock::DTCStringsBlock() 
  : m_pDTCStringBlock(new std::ostringstream())
{
  // Empty
}


DTCStringsBlock::~DTCStringsBlock() 
{
  if (m_pDTCStringBlock != NULL) {
    delete m_pDTCStringBlock;
    m_pDTCStringBlock = NULL;
  }
}

void
DTCStringsBlock::parseDTCStringsBlock(const char* _pBuffer, const unsigned int _size)
{

  XUtil::TRACE("Examining and extracting strings from the strings block image.");

  // Validate the buffer
  if (_pBuffer == NULL ) {
     throw std::runtime_error("ERROR: The given buffer pointer is NULL.");
  }

  if (_size == 0) {
    throw std::runtime_error("ERROR: The given buffer is empty.");
  }

  XUtil::TRACE_BUF("DTC Strings Block", _pBuffer, _size);

  if (_pBuffer[_size - 1] != 0) {
    throw std::runtime_error("Error: End of the strings block isn't terminated by null character.");
  }

  // Extract the data
  unsigned int index = 0;
  unsigned int lastIndex = 0;
  for (index = 0; index < _size; ++index) {
    if (_pBuffer[index] == 0) {
      int bufferLen = (index - lastIndex);
      std::string dtcString(&_pBuffer[lastIndex], bufferLen);

      XUtil::TRACE(XUtil::format("Adding DTCString: %s", dtcString.c_str()).c_str());
      unsigned int offset = addString(dtcString);
      
      if (offset != lastIndex) {
        std::string err = XUtil::format("ERROR: DTC string offset mismatch.  Expected: 0x%x, Actual: 0x%x", lastIndex, offset);
        throw std::runtime_error(err);
      }

      ++index;
      lastIndex = index;
    }
  }
}

unsigned int 
DTCStringsBlock::addString(const std::string _dtcString)
{
  // Has the string already been added
  std::string haystackString = m_pDTCStringBlock->str();
  std::size_t index = 0;

  // Create a new string that "includes" the null termination character
  std::string sNeedleString(_dtcString.c_str(), _dtcString.length()+ 1);
  index = haystackString.find(sNeedleString);

  // Did we find it?
  if (index != std::string::npos) {
    return (unsigned int) index;
  }

  // Not found, lets add it
  index = m_pDTCStringBlock->tellp();

  *m_pDTCStringBlock << _dtcString << '\0';
 
  return (unsigned int) index;
}


std::string  
DTCStringsBlock::getString(unsigned int _offset) const
{
  std::string blockString = m_pDTCStringBlock->str();

  if (_offset > blockString.size()) {
    std::string err = XUtil::format("ERROR: Offset (0x%x) is greater then the string buffer (0x%x).", _offset, blockString.size());
    throw std::runtime_error(err);
  }

  std::string returnString(blockString.c_str()+_offset);
  return returnString;
}


void 
DTCStringsBlock::marshalToDTC(std::ostream& _buf) const
{
  std::string blockString = m_pDTCStringBlock->str();
  _buf.write(blockString.c_str(), blockString.size());
}

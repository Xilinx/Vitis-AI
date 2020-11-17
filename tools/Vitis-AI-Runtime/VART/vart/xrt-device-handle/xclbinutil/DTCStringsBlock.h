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

#ifndef __DTCStringsBlock_h_
#define __DTCStringsBlock_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include <string>
#include <sstream>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ------------------- C L A S S :   S e c t i o n ---------------------------

class DTCStringsBlock {

 public:
  DTCStringsBlock();
  virtual ~DTCStringsBlock();

 public:
  unsigned int addString(const std::string _dtcString);
  std::string getString(unsigned int _offset) const;

  void parseDTCStringsBlock(const char* _pBuffer, const unsigned int _size);
  void marshalToDTC(std::ostream& _buf) const;

 private:
  // Purposefully private and undefined ctors...
  DTCStringsBlock(const DTCStringsBlock& obj);
  DTCStringsBlock& operator=(const DTCStringsBlock& obj);

 private:
   std::ostringstream * m_pDTCStringBlock;
};

#endif

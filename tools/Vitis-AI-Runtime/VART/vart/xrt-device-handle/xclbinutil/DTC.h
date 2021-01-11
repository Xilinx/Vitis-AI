/**
 * Copyright (C) 2019 Xilinx, Inc
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

#ifndef __DTC_h_
#define __DTC_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "DTCStringsBlock.h"
#include "FDTNode.h"

#include <boost/property_tree/ptree.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ------------------- C L A S S :   S e c t i o n ---------------------------

class DTC {

 public:
  DTC(const char* _pBuffer, const unsigned int _size, const FDTProperty::PropertyNameFormat & _propertyNameFormat);
  DTC(const boost::property_tree::ptree &_ptDTC, const FDTProperty::PropertyNameFormat & _propertyNameFormat);
  virtual ~DTC();

 public:
  void marshalToJSON(boost::property_tree::ptree &_dtcTree, const FDTProperty::PropertyNameFormat & _propertyNameFormat) const;
  void marshalToDTC(std::ostringstream& _buf) const;

 protected:
  void marshalFromDTCImage( const char* _pBuffer, const unsigned int _size, const FDTProperty::PropertyNameFormat & _propertyNameFormat);
  void marshalFromJSON(const boost::property_tree::ptree &_ptDTC, const FDTProperty::PropertyNameFormat & _propertyNameFormat);

 private:
  // Purposefully private and undefined ctors...
  DTC();
  DTC(const DTC& obj);
  DTC& operator=(const DTC& obj);

 private:
  DTCStringsBlock m_DTCStringsBlock;    // Block of property strings
  FDTNode * m_pTopFDTNode;              // Top FDTNode
};

#endif

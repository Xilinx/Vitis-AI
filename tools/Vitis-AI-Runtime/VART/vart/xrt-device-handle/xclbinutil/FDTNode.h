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

#ifndef __FDTNode_h_
#define __FDTNode_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "FDTProperty.h"

#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...
class DTCStringsBlock;

// ------------------- C L A S S :   S e c t i o n ---------------------------

class FDTNode {

 private:
  FDTNode(const char* _pBuffer, const unsigned int _size, const DTCStringsBlock& _dtcStringsBlock, unsigned int& _bytesExamined, const FDTProperty::PropertyNameFormat & _propertyNameFormat);
  FDTNode(const boost::property_tree::ptree& _ptDTC, std::string & _nodeName, const FDTProperty::PropertyNameFormat & _propertyNameFormat);

 public:
  virtual ~FDTNode();

 public:
  static FDTNode* marshalFromDTC(const char* _pBuffer, const unsigned int _size, const DTCStringsBlock& _dtcStringsBlock, const FDTProperty::PropertyNameFormat & _propertyNameFormat);
  static FDTNode* marshalFromJSON(const boost::property_tree::ptree& _ptDTC, const FDTProperty::PropertyNameFormat & _propertyNameFormat);

  void marshalToJSON(boost::property_tree::ptree& _dtcTree, const FDTProperty::PropertyNameFormat & _propertyNameFormat) const;
  void marshalToDTC(DTCStringsBlock& _dtcStringsBlock, std::ostream& _buf) const;

 protected:
  void marshalSubNodeToJSON(boost::property_tree::ptree& _ptTree, const FDTProperty::PropertyNameFormat & _propertyNameFormat) const;
  static void runningBufferCheck(const unsigned int _bytesExamined, const unsigned int _size);

 private:
  // Purposefully private and undefined ctors...
  FDTNode();
  FDTNode(const FDTNode& obj);
  FDTNode& operator=(const FDTNode& obj);

 private:
  std::string m_name;                      // Name of the node
  std::vector<FDTNode*> m_nestedNodes;     // Collection of nested nodes
  std::vector<FDTProperty*> m_properties;  // Node properties
};

#endif

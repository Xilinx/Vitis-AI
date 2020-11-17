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

#include "FDTNode.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

#include "FDTProperty.h"
#include "DTCStringsBlock.h"

#ifdef _WIN32
  #pragma comment(lib, "wsock32.lib")
  #include <winsock2.h>
#else
  #include <arpa/inet.h>
#endif


FDTNode::FDTNode() {
  // Empty
}


FDTNode::~FDTNode() 
{
  // Delete the pointers and clear the m_nestedNodes vector
  for (auto pNode : m_nestedNodes) {
    delete pNode;
  }
  m_nestedNodes.clear();

  // Delete the pointers and clear the m_properties vector
  for (auto pProperty : m_properties) {
    delete pProperty;
  }
  m_properties.clear();
}


void
FDTNode::runningBufferCheck(const unsigned int _bytesExamined, 
                            const unsigned int _size) 
{
  if (_bytesExamined > _size) {
    throw std::runtime_error("ERROR: Bytes examined exceeded size of buffer.");
  }
}


#define FDT_BEGIN_NODE  0x00000001
#define FDT_END_NODE    0x00000002
#define FDT_PROP        0x00000003
#define FDT_NOP         0x00000004
#define FDT_END         0x00000009


FDTNode::FDTNode(const char* _pBuffer,
                 const unsigned int _size,
                 const DTCStringsBlock& _dtcStringsBlock,
                 unsigned int& _bytesExamined,
                 const FDTProperty::PropertyNameFormat & _propertyNameFormat)
  : FDTNode() 
{
  XUtil::TRACE("Extracting FDT Node.");
  XUtil::TRACE_BUF("FDT Node Buffer", _pBuffer, _size);

  // Initialize return variables
  _bytesExamined = 0;

  // Validate the buffer
  if (_pBuffer == NULL) {
    throw std::runtime_error("ERROR: The given buffer pointer is NULL.");
  }

  if (_size == 0) {
    throw std::runtime_error("ERROR: The given buffer is empty.");
  }

  unsigned index = 0;                   // Running index
  // Get the node name
  {
    m_name = _pBuffer;                  // Get the null terminated ascii string
    index += (unsigned int) m_name.size() + 1;         // Count the "null" character
    runningBufferCheck(index, _size);

    XUtil::TRACE(XUtil::format("DTC Node Name: '%s'", m_name.c_str()));

    // Advance index to nearest 4 byte boundary
    if ((index % 4) != 0) {
      index += 3 - (index % 4);    // End of current word
      ++index;                     // Point to next word
    }
    runningBufferCheck(index, _size);
  }

  // Loop over the next group of tokens until an exit condition occurs
  while (true) {
    XUtil::TRACE(XUtil::format("Looping Index: %d (0x%x)", index, index));
    runningBufferCheck(index + sizeof(uint32_t), _size);
    uint32_t dtcToken = ntohl(*((const uint32_t*)&_pBuffer[index]));
    index += sizeof(uint32_t);

    // FDT_NOP
    switch (dtcToken) {
      case FDT_NOP:
        XUtil::TRACE("Token: FDT_NOP");
        continue;
        break;

      case FDT_BEGIN_NODE: {
          XUtil::TRACE("Token: FDT_BEGIN_NODE");
          const char* pBeginBuffer = &_pBuffer[index];
          unsigned int remainingSize = _size - index;
          unsigned int bytesExamined = 0;
          FDTNode* pNode = new FDTNode(pBeginBuffer, remainingSize, _dtcStringsBlock, bytesExamined, _propertyNameFormat);
          index += bytesExamined;
          runningBufferCheck(index, _size);
          m_nestedNodes.push_back(pNode);
          break;
        }

      case FDT_PROP: {
          XUtil::TRACE("Token: FDT_PROP");
          const char* pBeginBuffer = &_pBuffer[index];
          unsigned int remainingSize = _size - index;
          unsigned int bytesExamined = 0;
          FDTProperty* pProperty = new FDTProperty(pBeginBuffer, remainingSize, _dtcStringsBlock, bytesExamined, _propertyNameFormat);
          index += bytesExamined;
          runningBufferCheck(index, _size);
          m_properties.push_back(pProperty);
          break;
        }

      case FDT_END_NODE: {
          XUtil::TRACE("Token: FDT_END_NODE");
          _bytesExamined = index;
          return;             // <== Return from method
          break;
        }

      default: {
          std::string err = XUtil::format("ERROR: Unknown token: 0x%x", dtcToken);
          throw std::runtime_error(err);
        }
    }
  }
}

FDTNode::FDTNode(const boost::property_tree::ptree& _ptDTC, 
                 std::string & _nodeName,
                 const FDTProperty::PropertyNameFormat & _propertyNameFormat)
  : FDTNode() 
{
  m_name = _nodeName;

  // Iterator over the JSON element adding properties and subnodes

  for (boost::property_tree::ptree::const_iterator iter = _ptDTC.begin();
       iter != _ptDTC.end();
       ++iter) {
    std::string keyName = iter->first;

    if (_propertyNameFormat.find(keyName) != _propertyNameFormat.end()) {
      XUtil::TRACE(XUtil::format("Property Found: %s", keyName.c_str()).c_str());
      FDTProperty * pProperty = new FDTProperty(iter, _propertyNameFormat);
      m_properties.push_back(pProperty);
    } else {
      XUtil::TRACE(XUtil::format("Node Found: %s", keyName.c_str()).c_str());
      FDTNode * pFDTNode = new FDTNode(iter->second, keyName, _propertyNameFormat);
      m_nestedNodes.push_back(pFDTNode);
    }
  }
}


FDTNode*
FDTNode::marshalFromDTC( const char* _pBuffer, 
                         const unsigned int _size, 
                         const DTCStringsBlock& _dtcStringsBlock,
                         const FDTProperty::PropertyNameFormat & _propertyNameFormat) 
{
  XUtil::TRACE("Examining and extracting nodes from the structure block image.");

  // -- Validate the buffer and size --
  if (_pBuffer == NULL) {
    throw std::runtime_error("ERROR: The given buffer pointer is NULL.");
  }

  if (_size == 0) {
    throw std::runtime_error("ERROR: The given buffer is empty.");
  }

  XUtil::TRACE_BUF("Structure Block", _pBuffer, _size);

  const static unsigned int MinSizeBytes = sizeof(uint32_t) * 3; // 3 Words
  if (_size < MinSizeBytes) {
    std::string err = XUtil::format("ERROR: The size of the structure block is too small.  Minimum size: 0x%x", MinSizeBytes);
    throw std::runtime_error(err);
  }

  if ((_size % 4) != 0) {
    std::string err = XUtil::format("ERROR: The size of the structure block is not word aligned. Size: 0x%x", _size);
    throw std::runtime_error(err);
  }

  // -- Validate the first and and last tokens
  uint32_t startDtcToken = ntohl(*((const uint32_t*)&_pBuffer[0]));

  if (startDtcToken != FDT_BEGIN_NODE) {
    std::string err = XUtil::format("ERROR: Missing FDT_BEGIN_NODE token at the start of the structure block. Expected: 0x%x, Actual: 0x%x", FDT_BEGIN_NODE, startDtcToken);
    throw std::runtime_error(err);
  }

  uint32_t lastDtcToken = ntohl(*((const uint32_t*)&_pBuffer[(_size - sizeof(uint32_t))]));

  if (lastDtcToken != FDT_END) {
    std::string err = XUtil::format("ERROR: Missing FDT_END token at end of the structure block. Expected: 0x%x, Actual: 0x%x", FDT_END, lastDtcToken);
    throw std::runtime_error(err);
  }

  // -- Parse the nodes --
  unsigned index = 0;
  index += sizeof(uint32_t);    // Skip first token

  const char* pNodeBuffer = &_pBuffer[index];
  unsigned nodeBufferSize = _size - index;

  uint32_t bytesExamined = 0;
  FDTNode* pTopNode = new FDTNode(pNodeBuffer, nodeBufferSize, _dtcStringsBlock, bytesExamined, _propertyNameFormat);
  index += bytesExamined;
  runningBufferCheck(index, _size);

  uint32_t endDtcToken = ntohl(*((const uint32_t*)&_pBuffer[index]));
  index += sizeof(uint32_t);
  runningBufferCheck(index, _size);

  if (endDtcToken != FDT_END) {
    std::string err = XUtil::format("ERROR: Missing FDT_END_NODE token at end of the structure block. Expected: 0x%x, Actual: 0x%x", FDT_END, endDtcToken);
    throw std::runtime_error(err);
  }

  // -- Validate that all of the bytes were examined --
  if (_size != index) {
    std::string err = XUtil::format("ERROR: Structure Node Buffer wasn't completely examined.  Expected: 0x%x, Actual: 0x%s.", _size, index);
    throw std::runtime_error(err);
  }

  return pTopNode;
}



FDTNode*
FDTNode::marshalFromJSON( const boost::property_tree::ptree& _ptDTC,
                          const FDTProperty::PropertyNameFormat & _propertyNameFormat) 
{
  std::string topName = "";
  return new FDTNode(_ptDTC, topName, _propertyNameFormat);
}



void
FDTNode::marshalSubNodeToJSON(boost::property_tree::ptree& _ptTree, 
                              const FDTProperty::PropertyNameFormat & _propertyNameFormat) const 
{
  XUtil::TRACE(XUtil::format("** Examining SubNode: '%s'", m_name.c_str()));

//  if (m_name.length() == 0) {
//    std::string err = XUtil::format("ERROR: All nodes need a name.", m_name.c_str());
//    throw std::runtime_error(err);
//  }

  boost::property_tree::ptree ptSubNode;

  // Add the Properties
  for (auto pFDTProperty : m_properties) {
    pFDTProperty->marshalToJSON(ptSubNode, _propertyNameFormat);
  }

  // Add the underlying nodes
  for (auto pFDTNode : m_nestedNodes) {
    pFDTNode->marshalSubNodeToJSON(ptSubNode, _propertyNameFormat);
  }
  if (m_name.length() != 0) {
    _ptTree.add_child(m_name.c_str(), ptSubNode);
  } else {
    _ptTree.push_back(std::make_pair("", ptSubNode));
  }
}


void
FDTNode::marshalToJSON(boost::property_tree::ptree& _ptTree,
                       const FDTProperty::PropertyNameFormat & _propertyNameFormat) const 
{
  XUtil::TRACE(XUtil::format("** Examining Node: '%s'", m_name.c_str()));

  if (m_name.length() != 0) {
    std::string err = XUtil::format("ERROR: The given node '%s' is not the top node of the tree.", m_name.c_str());
    throw std::runtime_error(err);
  }

  // Add the Properties
  for (auto pFDTProperty : m_properties) {
    pFDTProperty->marshalToJSON(_ptTree, _propertyNameFormat);
  }

  // Add the underlying nodes
  for (auto pFDTNode : m_nestedNodes) {
    pFDTNode->marshalSubNodeToJSON(_ptTree, _propertyNameFormat);
  }
}


void
FDTNode::marshalToDTC(DTCStringsBlock& _dtcStringsBlock, std::ostream& _buf) const 
{
  // Add node keyword
  XUtil::write_htonl(_buf, FDT_BEGIN_NODE);

  // Add node name followed by null character
  _buf << m_name << '\0';

  // Pad if necessary
  XUtil::alignBytes(_buf, sizeof(uint32_t));

  // Write out the properties
  for (auto pFDTProperty : m_properties) {
    pFDTProperty->marshalToDTC(_dtcStringsBlock, _buf);
  }

  // Write out the nested nodes
  for (auto pFDTNode : m_nestedNodes) {
    pFDTNode->marshalToDTC(_dtcStringsBlock, _buf);
  }

  // Done with adding the node
  XUtil::write_htonl(_buf, FDT_END_NODE);
}

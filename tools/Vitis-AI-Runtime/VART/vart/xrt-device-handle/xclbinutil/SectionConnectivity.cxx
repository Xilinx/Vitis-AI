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

#include "SectionConnectivity.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;
#include <iostream>

// Static Variables / Classes
SectionConnectivity::_init SectionConnectivity::_initializer;

SectionConnectivity::SectionConnectivity() {
  // Empty
}

SectionConnectivity::~SectionConnectivity() {
  // Empty
}

void
SectionConnectivity::marshalToJSON(char* _pDataSection,
                                   unsigned int _sectionSize,
                                   boost::property_tree::ptree& _ptree) const {
  XUtil::TRACE("");
  XUtil::TRACE("Extracting: CONNECTIVITY");
  XUtil::TRACE_BUF("Section Buffer", reinterpret_cast<const char*>(_pDataSection), _sectionSize);

  // Do we have enough room to overlay the header structure
  if (_sectionSize < sizeof(connectivity)) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) is smaller than the size of the connectivity structure (%d)",
                                           _sectionSize, sizeof(connectivity)));
  }

  connectivity* pHdr = (connectivity*)_pDataSection;
  boost::property_tree::ptree connectivity;

  XUtil::TRACE(XUtil::format("m_count: %d", (unsigned int)pHdr->m_count));

  // Write out the entire structure except for the array structure
  XUtil::TRACE_BUF("connectivity", reinterpret_cast<const char*>(pHdr), ((uint64_t)&(pHdr->m_connection[0]) - (uint64_t)pHdr));
  connectivity.put("m_count", XUtil::format("%d", (unsigned int)pHdr->m_count).c_str());

  uint64_t expectedSize = ((uint64_t)&(pHdr->m_connection[0]) - (uint64_t)pHdr) + (sizeof(connection) * pHdr->m_count);

  if (_sectionSize != expectedSize) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) does not match expected section size (%d).",
                                           _sectionSize, expectedSize));
  }

  boost::property_tree::ptree m_connection;
  for (int index = 0; index < pHdr->m_count; ++index) {
    boost::property_tree::ptree connection;


    XUtil::TRACE(XUtil::format("[%d]: arg_index: %d, m_ip_layout_index: %d, mem_data_index: %d",
                               index,
                               (unsigned int)pHdr->m_connection[index].arg_index,
                               (unsigned int)pHdr->m_connection[index].m_ip_layout_index,
                               (unsigned int)pHdr->m_connection[index].mem_data_index));

    // Write out the entire structure
    XUtil::TRACE_BUF("connection", reinterpret_cast<const char*>(&(pHdr->m_connection[index])), sizeof(connection));

    connection.put("arg_index", XUtil::format("%d", (unsigned int)pHdr->m_connection[index].arg_index).c_str());
    connection.put("m_ip_layout_index", XUtil::format("%d", (unsigned int)pHdr->m_connection[index].m_ip_layout_index).c_str());
    connection.put("mem_data_index", XUtil::format("%d", (unsigned int)pHdr->m_connection[index].mem_data_index).c_str());

    m_connection.push_back(std::make_pair("", connection));   // Used to make an array of objects
  }

  connectivity.add_child("m_connection", m_connection);

  _ptree.add_child("connectivity", connectivity);
  XUtil::TRACE("-----------------------------");
}


void
SectionConnectivity::marshalFromJSON(const boost::property_tree::ptree& _ptSection,
                                     std::ostringstream& _buf) const {
  const boost::property_tree::ptree& ptConnectivity = _ptSection.get_child("connectivity");

  connectivity connectivityHdr = connectivity {0};

  // Read, store, and report mem_topology data
  connectivityHdr.m_count = ptConnectivity.get<uint32_t>("m_count");

  XUtil::TRACE("CONNECTIVITY");
  XUtil::TRACE(XUtil::format("m_count: %d", connectivityHdr.m_count));

  if (connectivityHdr.m_count == 0) {
    std::cout << "WARNING: Skipping CONNECTIVITY section for count size is zero." << std::endl;
    return;
  }

  // Write out the entire structure except for the mem_data structure
  XUtil::TRACE_BUF("connectivity - minus connection", reinterpret_cast<const char*>(&connectivityHdr), (sizeof(connectivity) - sizeof(connection)));
  _buf.write(reinterpret_cast<const char*>(&connectivityHdr), sizeof(connectivity) - sizeof(connection));


  // Read, store, and report connection segments
  unsigned int count = 0;
  const boost::property_tree::ptree connections = ptConnectivity.get_child("m_connection");
  for (const auto& kv : connections) {
    connection connectionHdr = connection {0};
    boost::property_tree::ptree ptConnection = kv.second;

    connectionHdr.arg_index = ptConnection.get<int32_t>("arg_index");
    connectionHdr.m_ip_layout_index = ptConnection.get<int32_t>("m_ip_layout_index");
    connectionHdr.mem_data_index = ptConnection.get<int32_t>("mem_data_index");

    XUtil::TRACE(XUtil::format("[%d]: arg_index: %d, m_ip_layout_index: %d, mem_data_index: %d",
                               count, (unsigned int)connectionHdr.arg_index,
                               (unsigned int)connectionHdr.m_ip_layout_index,
                               (unsigned int)connectionHdr.mem_data_index));

    // Write out the entire structure
    XUtil::TRACE_BUF("connection", reinterpret_cast<const char*>(&connectionHdr), sizeof(connection));
    _buf.write(reinterpret_cast<const char*>(&connectionHdr), sizeof(connection));
    count++;
  }

  // -- The counts should match --
  if (count != (unsigned int)connectivityHdr.m_count) {
    std::string errMsg = XUtil::format("ERROR: Number of connection sections (%d) does not match expected encoded value: %d",
                                       (unsigned int)count, (unsigned int)connectivityHdr.m_count);
    throw std::runtime_error(errMsg);
  }
  
  // -- Buffer needs to be less than 64K--
  unsigned int bufferSize = (unsigned int) _buf.str().size();
  const unsigned int maxBufferSize = 64 * 1024;
  if ( bufferSize > maxBufferSize ) {
    std::string errMsg = XUtil::format("CRITICAL WARNING: The buffer size for the CONNECTIVITY section (%d) exceed the maximum size of %d.\nThis can result in lose of data in the driver.",
                                       (unsigned int) bufferSize, (unsigned int) maxBufferSize);
    std::cout << errMsg << std::endl;
    // throw std::runtime_error(errMsg);
  }
}

bool 
SectionConnectivity::doesSupportAddFormatType(FormatType _eFormatType) const
{
  if (_eFormatType == FT_JSON) {
    return true;
  }
  return false;
}

bool 
SectionConnectivity::doesSupportDumpFormatType(FormatType _eFormatType) const
{
    if ((_eFormatType == FT_JSON) ||
        (_eFormatType == FT_HTML) ||
        (_eFormatType == FT_RAW))
    {
      return true;
    }

    return false;
}

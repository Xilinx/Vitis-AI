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

#include "SectionMemTopology.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;
#include <iostream>

// Static Variables / Classes
SectionMemTopology::_init SectionMemTopology::_initializer;

SectionMemTopology::SectionMemTopology() {
  // Empty
}

SectionMemTopology::~SectionMemTopology() {
  // Empty
}


const std::string
SectionMemTopology::getMemTypeStr(enum MEM_TYPE _memType) const {
  switch (_memType) {
    case MEM_DDR3:
      return "MEM_DDR3";
    case MEM_DDR4:
      return "MEM_DDR4";
    case MEM_DRAM:
      return "MEM_DRAM";
    case MEM_HBM:
      return "MEM_HBM";
    case MEM_BRAM:
      return "MEM_BRAM";
    case MEM_URAM:
      return "MEM_URAM";
    case MEM_STREAMING:
      return "MEM_STREAMING";
    case MEM_PREALLOCATED_GLOB:
      return "MEM_PREALLOCATED_GLOB";
    case MEM_ARE:
      return "MEM_ARE";
    case MEM_STREAMING_CONNECTION:
      return "MEM_STREAMING_CONNECTION";
  }

  return XUtil::format("UNKNOWN (%d)", (unsigned int)_memType);
}

enum MEM_TYPE
SectionMemTopology::getMemType(std::string& _sMemType) const {
  if (_sMemType == "MEM_DDR3")
    return MEM_DDR3;

  if (_sMemType == "MEM_DDR4")
    return MEM_DDR4;

  if (_sMemType == "MEM_DRAM")
    return MEM_DRAM;

  if (_sMemType == "MEM_HBM")
    return MEM_HBM;

  if (_sMemType == "MEM_BRAM")
    return MEM_BRAM;

  if (_sMemType == "MEM_URAM")
    return MEM_URAM;

  if (_sMemType == "MEM_STREAMING")
    return MEM_STREAMING;

  if (_sMemType == "MEM_PREALLOCATED_GLOB")
    return MEM_PREALLOCATED_GLOB;

  if (_sMemType == "MEM_ARE")
    return MEM_ARE;

  if (_sMemType == "MEM_STREAMING_CONNECTION")
    return MEM_STREAMING_CONNECTION;

  std::string errMsg = "ERROR: Unknown memory type: '" + _sMemType + "'";
  throw std::runtime_error(errMsg);
}


void
SectionMemTopology::marshalToJSON(char* _pDataSection,
                                  unsigned int _sectionSize,
                                  boost::property_tree::ptree& _ptree) const {
  XUtil::TRACE("");
  XUtil::TRACE("Extracting: MEM_TOPOLOGY");
  XUtil::TRACE_BUF("Section Buffer", reinterpret_cast<const char*>(_pDataSection), _sectionSize);

  // Do we have enough room to overlay the header structure
  if (_sectionSize < sizeof(mem_topology)) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) is smaller than the size of the mem_topology structure (%d)",
                                           _sectionSize, sizeof(mem_topology)));
  }

  mem_topology* pHdr = (mem_topology*)_pDataSection;
  boost::property_tree::ptree mem_topology;

  XUtil::TRACE(XUtil::format("m_count: %d", pHdr->m_count));

  // Write out the entire structure except for the array structure
  XUtil::TRACE_BUF("mem_topology", reinterpret_cast<const char*>(pHdr), ((uint64_t)&(pHdr->m_mem_data[0]) - (uint64_t)pHdr));
  mem_topology.put("m_count", XUtil::format("%d", (unsigned int)pHdr->m_count).c_str());

  uint64_t expectedSize = ((uint64_t)&(pHdr->m_mem_data[0]) - (uint64_t)pHdr) + (sizeof(mem_data) * pHdr->m_count);

  if (_sectionSize != expectedSize) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) does not match expected section size (%d).",
                                           _sectionSize, expectedSize));
  }

  boost::property_tree::ptree m_mem_data;
  for (int index = 0; index < pHdr->m_count; ++index) {
    boost::property_tree::ptree mem_data;

    XUtil::TRACE(XUtil::format("[%d]: m_type: %s, m_used: %d, m_sizeKB: 0x%lx, m_tag: '%s', m_base_address: 0x%lx",
                               index,
                               getMemTypeStr((enum MEM_TYPE)pHdr->m_mem_data[index].m_type).c_str(),
                               (unsigned int)pHdr->m_mem_data[index].m_used,
                               pHdr->m_mem_data[index].m_size,
                               pHdr->m_mem_data[index].m_tag,
                               pHdr->m_mem_data[index].m_base_address));

    // Write out the entire structure
    XUtil::TRACE_BUF("mem_data", reinterpret_cast<const char*>(&(pHdr->m_mem_data[index])), sizeof(mem_data));

    mem_data.put("m_type", getMemTypeStr((enum MEM_TYPE)pHdr->m_mem_data[index].m_type).c_str());
    mem_data.put("m_used", XUtil::format("%d", (unsigned int)pHdr->m_mem_data[index].m_used).c_str());
    mem_data.put("m_sizeKB", XUtil::format("0x%lx", pHdr->m_mem_data[index].m_size).c_str());
    mem_data.put("m_tag", XUtil::format("%s", pHdr->m_mem_data[index].m_tag).c_str());
    mem_data.put("m_base_address", XUtil::format("0x%lx", pHdr->m_mem_data[index].m_base_address).c_str());

    m_mem_data.push_back(std::make_pair("", mem_data));   // Used to make an array of objects
  }

  mem_topology.add_child("m_mem_data", m_mem_data);

  _ptree.add_child("mem_topology", mem_topology);
  XUtil::TRACE("-----------------------------");
}


void
SectionMemTopology::marshalFromJSON(const boost::property_tree::ptree& _ptSection,
                                    std::ostringstream& _buf) const {
  const boost::property_tree::ptree& ptMemtopPayload = _ptSection.get_child("mem_topology");

  mem_topology memTopologyHdr = mem_topology {0};

  // Read, store, and report mem_topology data
  memTopologyHdr.m_count = ptMemtopPayload.get<uint32_t>("m_count");

  XUtil::TRACE("MEM_TOPOLOGY");
  XUtil::TRACE(XUtil::format("m_count: %d", memTopologyHdr.m_count));

  if (memTopologyHdr.m_count == 0) {
    std::cout << "WARNING: Skipping CONNECTIVITY section for count size is zero." << std::endl;
    return;
  }

  // Write out the entire structure except for the mem_data structure
  XUtil::TRACE_BUF("mem_topology - minus mem_data", reinterpret_cast<const char*>(&memTopologyHdr), (sizeof(mem_topology) - sizeof(mem_data)));
  _buf.write(reinterpret_cast<const char*>(&memTopologyHdr), (sizeof(mem_topology) - sizeof(mem_data)));


  // Read, store, and report mem_data segments
  unsigned int count = 0;
  boost::property_tree::ptree memDatas = ptMemtopPayload.get_child("m_mem_data");
  for (const auto& kv : memDatas) {
    mem_data memData = mem_data {0};
    boost::property_tree::ptree ptMemData = kv.second;

    std::string sm_type = ptMemData.get<std::string>("m_type");
    memData.m_type = (uint8_t)getMemType(sm_type);

    memData.m_used = ptMemData.get<uint8_t>("m_used");

    std::string sm_tag = ptMemData.get<std::string>("m_tag");
    if (sm_tag.length() >= sizeof(mem_data::m_tag)) {
      std::string errMsg = XUtil::format("ERROR: The m_tag entry length (%d), exceeds the allocated space (%d).  Name: '%s'",
                                         (unsigned int)sm_tag.length(), (unsigned int)sizeof(mem_data::m_tag), sm_tag.c_str());
      throw std::runtime_error(errMsg);
    }

    // We already know that there is enough room for this string
    memcpy(memData.m_tag, sm_tag.c_str(), sm_tag.length() + 1);

    // No more data to read in for the MEM_STREAMING_CONNECTION type.
    // Note: This data structure is initialized with zeros.
    if (memData.m_type != MEM_STREAMING_CONNECTION) {

      boost::optional<std::string> sizeBytes = ptMemData.get_optional<std::string>("m_size");

      if (sizeBytes.is_initialized()) {
        memData.m_size = XUtil::stringToUInt64(static_cast<std::string>(sizeBytes.get()));
        if ((memData.m_size % 1024) != 0)
          throw std::runtime_error(XUtil::format("ERROR: The memory size (%ld) does not align to a 1K (1024 bytes) boundary.", memData.m_size));

        memData.m_size = memData.m_size / (uint64_t)1024;
      }

      boost::optional<std::string> sizeKB = ptMemData.get_optional<std::string>("m_sizeKB");
      if (sizeBytes.is_initialized() && sizeKB.is_initialized())
        throw std::runtime_error(XUtil::format("ERROR: 'm_size' (%s) and 'm_sizeKB' (%s) are mutually exclusive.",
                                               static_cast<std::string>(sizeBytes.get()).c_str(),
                                               static_cast<std::string>(sizeKB.get()).c_str()));

      if (sizeKB.is_initialized())
        memData.m_size = XUtil::stringToUInt64(static_cast<std::string>(sizeKB.get()));


      std::string sBaseAddress = ptMemData.get<std::string>("m_base_address");
      memData.m_base_address = XUtil::stringToUInt64(sBaseAddress);
    }

    XUtil::TRACE(XUtil::format("[%d]: m_type: %d, m_used: %d, m_size: 0x%lx, m_tag: '%s', m_base_address: 0x%lx",
                               count,
                               (unsigned int)memData.m_type,
                               (unsigned int)memData.m_used,
                               memData.m_size,
                               memData.m_tag,
                               memData.m_base_address));

    // Write out the entire structure
    XUtil::TRACE_BUF("mem_data", reinterpret_cast<const char*>(&memData), sizeof(mem_data));
    _buf.write(reinterpret_cast<const char*>(&memData), sizeof(mem_data));
    count++;
  }

  // -- The counts should match --
  if (count != (unsigned int)memTopologyHdr.m_count) {
    std::string errMsg = XUtil::format("ERROR: Number of mem_data sections (%d) does not match expected encoded value: %d",
                                       (unsigned int)count, (unsigned int)memTopologyHdr.m_count);
    throw std::runtime_error(errMsg);
  }

  // -- Buffer needs to be less than 64K--
  unsigned int bufferSize = (unsigned int) _buf.str().size();
  const unsigned int maxBufferSize = 64 * 1024;
  if ( bufferSize > maxBufferSize ) {
    std::string errMsg = XUtil::format("CRITICAL WARNING: The buffer size for the MEM_TOPOLOGY (%d) exceed the maximum size of %d.\nThis can result in lose of data in the driver.",
                                       (unsigned int) bufferSize, (unsigned int) maxBufferSize);
    std::cout << errMsg << std::endl;
    // throw std::runtime_error(errMsg);
  }
}

bool 
SectionMemTopology::doesSupportAddFormatType(FormatType _eFormatType) const
{
  if (_eFormatType == FT_JSON) {
    return true;
  }
  return false;
}

bool 
SectionMemTopology::doesSupportDumpFormatType(FormatType _eFormatType) const
{
    if ((_eFormatType == FT_JSON) ||
        (_eFormatType == FT_HTML) ||
        (_eFormatType == FT_RAW))
    {
      return true;
    }

    return false;
}

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

#include "SectionClockFrequencyTopology.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;
#include <iostream>
#include <stdint.h>

// Static Variables / Classes
SectionClockFrequencyTopology::_init SectionClockFrequencyTopology::_initializer;

SectionClockFrequencyTopology::SectionClockFrequencyTopology() {
  // Empty
}

SectionClockFrequencyTopology::~SectionClockFrequencyTopology() {
  // Empty
}

const std::string
SectionClockFrequencyTopology::getClockTypeStr(enum CLOCK_TYPE _clockType) const {
  switch (_clockType) {
    case CT_UNUSED:
      return "UNUSED";
    case CT_DATA:
      return "DATA";
    case CT_KERNEL:
      return "KERNEL";
    case CT_SYSTEM:
      return "SYSTEM";
  }

  return XUtil::format("UNKNOWN (%d) CLOCK_TYPE", (unsigned int)_clockType);
}

enum CLOCK_TYPE
SectionClockFrequencyTopology::getClockType(std::string& _sClockType) const {
  if (_sClockType == "UNUSED")
    return CT_UNUSED;

  if (_sClockType == "DATA")
    return CT_DATA;

  if (_sClockType == "KERNEL")
    return CT_KERNEL;

  if (_sClockType == "SYSTEM")
    return CT_SYSTEM;

  std::string errMsg = "ERROR: Unknown Clock Type: '" + _sClockType + "'";
  throw std::runtime_error(errMsg);
}


void
SectionClockFrequencyTopology::marshalToJSON(char* _pDataSection,
                                             unsigned int _sectionSize,
                                             boost::property_tree::ptree& _ptree) const {
  XUtil::TRACE("");
  XUtil::TRACE("Marshalling to JSON: ClockFreqTopology");
  XUtil::TRACE_BUF("Section Buffer", reinterpret_cast<const char*>(_pDataSection), _sectionSize);

  // Do we have enough room to overlay the header structure
  if (_sectionSize < sizeof(clock_freq_topology)) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) is smaller than the size of the clock_freq_topology structure (%d)",
                                           _sectionSize, sizeof(clock_freq_topology)));
  }

  clock_freq_topology* pHdr = (clock_freq_topology*)_pDataSection;
  boost::property_tree::ptree clock_freq_topology;

  XUtil::TRACE(XUtil::format("m_count: %d", (uint32_t)pHdr->m_count));

  // Write out the entire structure except for the array structure
  XUtil::TRACE_BUF("clock_freq", reinterpret_cast<const char*>(pHdr), ((uint64_t)&(pHdr->m_clock_freq[0]) - (uint64_t) pHdr));
  clock_freq_topology.put("m_count", XUtil::format("%d", (unsigned int)pHdr->m_count).c_str());

  clock_freq mydata = clock_freq {0};

  XUtil::TRACE(XUtil::format("Size of clock_freq: %d\nSize of mydata: %d",
                             sizeof(clock_freq),
                             sizeof(mydata)));
  uint64_t expectedSize = ((uint64_t)&(pHdr->m_clock_freq[0]) - (uint64_t) pHdr) + (sizeof(clock_freq) * (uint64_t)pHdr->m_count);

  if (_sectionSize != expectedSize) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) does not match expected sections size (%d).",
                                           _sectionSize, expectedSize));
  }

  boost::property_tree::ptree m_clock_freq;
  for (int index = 0; index < pHdr->m_count; ++index) {
    boost::property_tree::ptree clock_freq;

    XUtil::TRACE(XUtil::format("[%d]: m_freq_Mhz: %d, m_type: %d, m_name: '%s'",
                               index,
                               (unsigned int)pHdr->m_clock_freq[index].m_freq_Mhz,
                               getClockTypeStr((enum CLOCK_TYPE)pHdr->m_clock_freq[index].m_type).c_str(),
                               pHdr->m_clock_freq[index].m_name));

    // Write out the entire structure
    XUtil::TRACE_BUF("clock_freq", reinterpret_cast<const char*>(&pHdr->m_clock_freq[index]), sizeof(clock_freq));

    clock_freq.put("m_freq_Mhz", XUtil::format("%d", (unsigned int)pHdr->m_clock_freq[index].m_freq_Mhz).c_str());
    clock_freq.put("m_type", getClockTypeStr((enum CLOCK_TYPE)pHdr->m_clock_freq[index].m_type).c_str());
    clock_freq.put("m_name", XUtil::format("%s", pHdr->m_clock_freq[index].m_name).c_str());

    m_clock_freq.push_back(std::make_pair("", clock_freq));   // Used to make an array of objects
  }

  clock_freq_topology.add_child("m_clock_freq", m_clock_freq);

  _ptree.add_child("clock_freq_topology", clock_freq_topology);
  XUtil::TRACE("-----------------------------");
}



void
SectionClockFrequencyTopology::marshalFromJSON(const boost::property_tree::ptree& _ptSection,
                                               std::ostringstream& _buf) const {
  const boost::property_tree::ptree& ptClockFreqTopo = _ptSection.get_child("clock_freq_topology");

  // Initialize the memory to zero's
  clock_freq_topology clockFreqTopologyHdr = clock_freq_topology {0};

  // Read, store, and report clock frequency topology data
  clockFreqTopologyHdr.m_count = ptClockFreqTopo.get<uint16_t>("m_count");

  XUtil::TRACE("CLOCK_FREQ_TOPOLOGY");
  XUtil::TRACE(XUtil::format("m_count: %d", clockFreqTopologyHdr.m_count));

  if (clockFreqTopologyHdr.m_count == 0) {
    std::cout << "WARNING: Skipping CLOCK_FREQ_TOPOLOGY section for count size is zero." << std::endl;
    return;
  }

  // Write out the entire structure except for the mem_data structure
  XUtil::TRACE_BUF("clock_freq_topology- minus clock_freq", reinterpret_cast<const char*>(&clockFreqTopologyHdr), (sizeof(clock_freq_topology) - sizeof(clock_freq)));
  _buf.write(reinterpret_cast<const char*>(&clockFreqTopologyHdr), sizeof(clock_freq_topology) - sizeof(clock_freq));

  // Read, store, and report connection sections
  unsigned int count = 0;
  const boost::property_tree::ptree clockFreqs = ptClockFreqTopo.get_child("m_clock_freq");
  for (const auto& kv : clockFreqs) {
    clock_freq clockFreqHdr = clock_freq {0};
    boost::property_tree::ptree ptClockFreq = kv.second;

    clockFreqHdr.m_freq_Mhz = ptClockFreq.get<uint16_t>("m_freq_Mhz");
    std::string sm_type = ptClockFreq.get<std::string>("m_type");
    clockFreqHdr.m_type = (uint8_t) getClockType(sm_type);

    std::string sm_name = ptClockFreq.get<std::string>("m_name");
    if (sm_name.length() >= sizeof(clock_freq::m_name)) {
      std::string errMsg = XUtil::format("ERROR: The m_name entry length (%d), exceeds the allocated space (%d).  Name: '%s'",
                                         (unsigned int)sm_name.length(), (unsigned int)sizeof(clock_freq::m_name), sm_name.c_str());
      throw std::runtime_error(errMsg);
    }

    // We already know that there is enough room for this string
    memcpy(clockFreqHdr.m_name, sm_name.c_str(), sm_name.length() + 1);

    XUtil::TRACE(XUtil::format("[%d]: m_freq_Mhz: %d, m_type: %d, m_name: '%s'",
                               count,
                               (unsigned int)clockFreqHdr.m_freq_Mhz,
                               (unsigned int)clockFreqHdr.m_type,
                               clockFreqHdr.m_name));

    // Write out the entire structure
    XUtil::TRACE_BUF("clock_freq", reinterpret_cast<const char*>(&clockFreqHdr), sizeof(clock_freq));
    _buf.write(reinterpret_cast<const char*>(&clockFreqHdr), sizeof(clock_freq));
    count++;
  }

  // -- The counts should match --
  if (count != (unsigned int)clockFreqTopologyHdr.m_count) {
    std::string errMsg = XUtil::format("ERROR: Number of connection sections (%d) does not match expected encoded value: %d",
                                       (unsigned int)count, (unsigned int)clockFreqTopologyHdr.m_count);
    throw std::runtime_error(errMsg);
  }
}

bool 
SectionClockFrequencyTopology::doesSupportAddFormatType(FormatType _eFormatType) const
{
  if (_eFormatType == FT_JSON) {
    return true;
  }
  return false;
}

bool 
SectionClockFrequencyTopology::doesSupportDumpFormatType(FormatType _eFormatType) const
{
    if ((_eFormatType == FT_JSON) ||
        (_eFormatType == FT_HTML) ||
        (_eFormatType == FT_RAW))
    {
      return true;
    }

    return false;
}

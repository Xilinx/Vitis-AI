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

#include "SectionIPLayout.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

#include <iostream>

// Static Variables / Classes
SectionIPLayout::_init SectionIPLayout::_initializer;

SectionIPLayout::SectionIPLayout() {
  // Empty
}

SectionIPLayout::~SectionIPLayout() {
  // Empty
}


const std::string
SectionIPLayout::getIPTypeStr(enum IP_TYPE _ipType) const {
  switch (_ipType) {
    case IP_MB:
      return "IP_MB";
    case IP_KERNEL:
      return "IP_KERNEL";
    case IP_DNASC:
      return "IP_DNASC";
    case IP_DDR4_CONTROLLER:
      return "IP_DDR4_CONTROLLER";
    case IP_MEM_DDR4:
      return "IP_MEM_DDR4";
    case IP_MEM_HBM:
      return "IP_MEM_HBM";
  }

  return XUtil::format("UNKNOWN (%d)", (unsigned int)_ipType);
}

enum IP_TYPE
SectionIPLayout::getIPType(std::string& _sIPType) const {
  if (_sIPType == "IP_MB") return IP_MB;
  if (_sIPType == "IP_KERNEL") return IP_KERNEL;
  if (_sIPType == "IP_DNASC") return IP_DNASC;
  if (_sIPType == "IP_DDR4_CONTROLLER") return IP_DDR4_CONTROLLER;
  if (_sIPType == "IP_MEM_DDR4") return IP_MEM_DDR4;
  if (_sIPType == "IP_MEM_HBM") return IP_MEM_HBM;

  std::string errMsg = "ERROR: Unknown IP type: '" + _sIPType + "'";
  throw std::runtime_error(errMsg);
}

const std::string
SectionIPLayout::getIPControlTypeStr(enum IP_CONTROL _ipControlType) const {
  switch (_ipControlType) {
    case AP_CTRL_HS:
      return "AP_CTRL_HS";
    case AP_CTRL_CHAIN:
      return "AP_CTRL_CHAIN";
    case AP_CTRL_ME:
      return "AP_CTRL_ME";
    case AP_CTRL_NONE:
      return "AP_CTRL_NONE";
    case ACCEL_ADAPTER:
      return "ACCEL_ADAPTER";
    default:
      return "";
  }

  return XUtil::format("UNKNOWN (%d)", (unsigned int) _ipControlType);
}


enum IP_CONTROL
SectionIPLayout::getIPControlType(std::string& _sIPControlType) const {
  if (_sIPControlType == "AP_CTRL_HS") return AP_CTRL_HS;
  if (_sIPControlType == "AP_CTRL_CHAIN") return AP_CTRL_CHAIN;
  if (_sIPControlType == "AP_CTRL_ME") return AP_CTRL_ME;
  if (_sIPControlType == "AP_CTRL_NONE") return AP_CTRL_NONE;
  if (_sIPControlType == "ACCEL_ADAPTER") return ACCEL_ADAPTER;

  std::string errMsg = "ERROR: Unknown IP Control type: '" + _sIPControlType + "'";
  throw std::runtime_error(errMsg);
}


void
SectionIPLayout::marshalToJSON(char* _pDataSection,
                               unsigned int _sectionSize,
                               boost::property_tree::ptree& _ptree) const {
  XUtil::TRACE("");
  XUtil::TRACE("Extracting: IP_LAYOUT");
  XUtil::TRACE_BUF("Section Buffer", reinterpret_cast<const char*>(_pDataSection), _sectionSize);

  // Do we have enough room to overlay the header structure
  if (_sectionSize < sizeof(ip_layout)) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) is smaller than the size of the ip_layout structure (%d)",
                                           _sectionSize, sizeof(ip_layout)));
  }

  ip_layout* pHdr = (ip_layout*)_pDataSection;
  boost::property_tree::ptree ip_layout;

  XUtil::TRACE(XUtil::format("m_count: %d", pHdr->m_count));

  // Write out the entire structure except for the array structure
  XUtil::TRACE_BUF("ip_layout", reinterpret_cast<const char*>(pHdr), ((uint64_t)&(pHdr->m_ip_data[0]) - (uint64_t)pHdr));
  ip_layout.put("m_count", XUtil::format("%d", (unsigned int)pHdr->m_count).c_str());

  uint64_t expectedSize = ((uint64_t)&(pHdr->m_ip_data[0]) - (uint64_t)pHdr) + (sizeof(ip_data) * pHdr->m_count);

  if (_sectionSize != expectedSize) {
    throw std::runtime_error(XUtil::format("ERROR: Section size (%d) does not match expected section size (%d).",
                                           _sectionSize, expectedSize));
  }

  boost::property_tree::ptree m_ip_data;
  for (int index = 0; index < pHdr->m_count; ++index) {
    boost::property_tree::ptree ip_data;

    if (((enum IP_TYPE)pHdr->m_ip_data[index].m_type == IP_MEM_DDR4) ||
        ((enum IP_TYPE)pHdr->m_ip_data[index].m_type == IP_MEM_HBM)) {

      XUtil::TRACE(XUtil::format("[%d]: m_type: %s, m_index: %d, m_pc_index: %d, m_base_address: 0x%lx, m_name: '%s'",
                                 index,
                                 getIPTypeStr((enum IP_TYPE)pHdr->m_ip_data[index].m_type).c_str(),
                                 pHdr->m_ip_data[index].indices.m_index,
                                 pHdr->m_ip_data[index].indices.m_pc_index,
                                 pHdr->m_ip_data[index].m_base_address,
                                 pHdr->m_ip_data[index].m_name));
    } else if ((enum IP_TYPE)pHdr->m_ip_data[index].m_type == IP_KERNEL) {
      std::string sIPControlType = getIPControlTypeStr((enum IP_CONTROL) ((pHdr->m_ip_data[index].properties & ((uint32_t) IP_CONTROL_MASK)) >> IP_CONTROL_SHIFT));
      XUtil::TRACE(XUtil::format("[%d]: m_type: %s, properties: 0x%x {m_ip_control: %s, m_interrupt_id: %d, m_int_enable: %d}, m_base_address: 0x%lx, m_name: '%s'",
                                 index,
                                 getIPTypeStr((enum IP_TYPE)pHdr->m_ip_data[index].m_type).c_str(),
                                 pHdr->m_ip_data[index].properties,
                                 sIPControlType.c_str(),
                                 (pHdr->m_ip_data[index].properties & ((uint32_t) IP_INTERRUPT_ID_MASK)) >> IP_INTERRUPT_ID_SHIFT,
                                 (pHdr->m_ip_data[index].properties & ((uint32_t) IP_INT_ENABLE_MASK)),
                                 pHdr->m_ip_data[index].m_base_address,
                                 pHdr->m_ip_data[index].m_name));
    } else {
      XUtil::TRACE(XUtil::format("[%d]: m_type: %s, properties: 0x%x, m_base_address: 0x%lx, m_name: '%s'",
                                 index,
                                 getIPTypeStr((enum IP_TYPE)pHdr->m_ip_data[index].m_type).c_str(),
                                 pHdr->m_ip_data[index].properties,
                                 pHdr->m_ip_data[index].m_base_address,
                                 pHdr->m_ip_data[index].m_name));
    }

    // Write out the entire structure
    XUtil::TRACE_BUF("ip_data", reinterpret_cast<const char*>(&(pHdr->m_ip_data[index])), sizeof(ip_data));

    ip_data.put("m_type", getIPTypeStr((enum IP_TYPE)pHdr->m_ip_data[index].m_type).c_str());

    if (((enum IP_TYPE)pHdr->m_ip_data[index].m_type == IP_MEM_DDR4) ||
        ((enum IP_TYPE)pHdr->m_ip_data[index].m_type == IP_MEM_HBM)) {
      ip_data.put("m_index", XUtil::format("%d", (unsigned int)pHdr->m_ip_data[index].indices.m_index).c_str());
      ip_data.put("m_pc_index", XUtil::format("%d", (unsigned int)pHdr->m_ip_data[index].indices.m_pc_index).c_str());
    } else if ((enum IP_TYPE)pHdr->m_ip_data[index].m_type == IP_KERNEL) {
      ip_data.put("m_int_enable", XUtil::format("%d", (pHdr->m_ip_data[index].properties & ((uint32_t) IP_INT_ENABLE_MASK))).c_str());
      ip_data.put("m_interrupt_id", XUtil::format("%d", (pHdr->m_ip_data[index].properties & ((uint32_t) IP_INTERRUPT_ID_MASK)) >> IP_INTERRUPT_ID_SHIFT).c_str());
      std::string sIPControlType = getIPControlTypeStr((enum IP_CONTROL) ((pHdr->m_ip_data[index].properties & ((uint32_t) IP_CONTROL_MASK)) >> IP_CONTROL_SHIFT));
      ip_data.put("m_ip_control", sIPControlType.c_str());
    } else {
      ip_data.put("properties", XUtil::format("0x%x", pHdr->m_ip_data[index].properties).c_str());
    }
    if ( pHdr->m_ip_data[index].m_base_address != ((uint64_t) -1) ) {
      ip_data.put("m_base_address", XUtil::format("0x%lx", pHdr->m_ip_data[index].m_base_address).c_str());
    } else {
      ip_data.put("m_base_address", "not_used");
    }
    ip_data.put("m_name", XUtil::format("%s", pHdr->m_ip_data[index].m_name).c_str());

    m_ip_data.push_back(std::make_pair("", ip_data));   // Used to make an array of objects
  }

  ip_layout.add_child("m_ip_data", m_ip_data);

  _ptree.add_child("ip_layout", ip_layout);
  XUtil::TRACE("-----------------------------");
}

void
SectionIPLayout::marshalFromJSON(const boost::property_tree::ptree& _ptSection,
                                 std::ostringstream& _buf) const {
  const boost::property_tree::ptree& ptIPLayout = _ptSection.get_child("ip_layout");

  // Initialize the memory to zero's
  ip_layout ipLayoutHdr = ip_layout {0};

  // Read, store, and report mem_topology data
  ipLayoutHdr.m_count = ptIPLayout.get<uint32_t>("m_count");

  if (ipLayoutHdr.m_count == 0) {
    std::cout << "WARNING: Skipping IP_LAYOUT section for count size is zero." << std::endl;
    return;
  }

  XUtil::TRACE("IP_LAYOUT");
  XUtil::TRACE(XUtil::format("m_count: %d", ipLayoutHdr.m_count));

  // Write out the entire structure except for the mem_data structure
  XUtil::TRACE_BUF("ip_layout - minus ip_data", reinterpret_cast<const char*>(&ipLayoutHdr), (sizeof(ip_layout) - sizeof(ip_data)));
  _buf.write(reinterpret_cast<const char*>(&ipLayoutHdr), sizeof(ip_layout) - sizeof(ip_data));


  // Read, store, and report connection segments
  unsigned int count = 0;
  boost::property_tree::ptree ipDatas = ptIPLayout.get_child("m_ip_data");
  for (const auto& kv : ipDatas) {
    ip_data ipDataHdr = ip_data {0};
    boost::property_tree::ptree ptIPData = kv.second;

    std::string sm_type = ptIPData.get<std::string>("m_type");
    ipDataHdr.m_type = getIPType(sm_type);

    // For these IPs, the struct indices needs to be initialized
    if ((ipDataHdr.m_type == IP_MEM_DDR4) ||
        (ipDataHdr.m_type == IP_MEM_HBM))
    {
      ipDataHdr.indices.m_index = ptIPData.get<uint16_t>("m_index");
      ipDataHdr.indices.m_pc_index = ptIPData.get<uint8_t>("m_pc_index", 0);
    } else {
      // Get the properties value (if one is defined)
      std::string sProperties = ptIPData.get<std::string>("properties", "0");
      ipDataHdr.properties = (uint32_t)XUtil::stringToUInt64(sProperties);
      
      { // m_int_enable
        boost::optional<bool> bIntEnable;
        bIntEnable = ptIPData.get_optional<bool>("m_int_enable");
        if (bIntEnable.is_initialized()) {
          ipDataHdr.properties = ipDataHdr.properties & (~(uint32_t) IP_INT_ENABLE_MASK);  // Clear existing bit
          if (bIntEnable.get()) {
            ipDataHdr.properties = ipDataHdr.properties | ((uint32_t) IP_INT_ENABLE_MASK); // Set bit
          }
        }
      }
  
      { // m_interrupt_id
        boost::optional<uint8_t> bInterruptID;
        bInterruptID = ptIPData.get_optional<uint8_t>("m_interrupt_id");
        if (bInterruptID.is_initialized()) {
          unsigned int maxValue = ((unsigned int) IP_INTERRUPT_ID_MASK) >> IP_INTERRUPT_ID_SHIFT;
          if (bInterruptID.get() > maxValue) {
            std::string errMsg = XUtil::format("ERROR: The m_interrupt_id (%d), exceeds maximum value (%d).",
                                               (unsigned int)bInterruptID.get(), maxValue);
            throw std::runtime_error(errMsg);
          }
  
          unsigned int shiftValue = ((unsigned int) bInterruptID.get()) << IP_INTERRUPT_ID_SHIFT;
          shiftValue = shiftValue & ((uint32_t) IP_INTERRUPT_ID_MASK);
          ipDataHdr.properties = ipDataHdr.properties & (~(uint32_t) IP_INTERRUPT_ID_MASK);  // Clear existing bits
          ipDataHdr.properties = ipDataHdr.properties | shiftValue;                          // Set bits
        }
      }
  
      { // m_ip_control
        boost::optional<std::string> bIPControl;
        bIPControl = ptIPData.get_optional<std::string>("m_ip_control");
        if (bIPControl.is_initialized()) {
          unsigned int ipControl = (unsigned int) getIPControlType(bIPControl.get());
  
          unsigned int maxValue = ((unsigned int) IP_CONTROL_MASK) >> IP_CONTROL_SHIFT;
          if (ipControl > maxValue) {
            std::string errMsg = XUtil::format("ERROR: The m_ip_control (%d), exceeds maximum value (%d).",
                                               (unsigned int) ipControl, maxValue);
            throw std::runtime_error(errMsg);
          }
  
          unsigned int shiftValue = ipControl << IP_CONTROL_SHIFT;
          shiftValue = shiftValue & ((uint32_t) IP_CONTROL_MASK);
          ipDataHdr.properties = ipDataHdr.properties & (~(uint32_t) IP_CONTROL_MASK);  // Clear existing bits
          ipDataHdr.properties = ipDataHdr.properties | shiftValue;                          // Set bits
        }
      }
    }

    std::string sBaseAddress = ptIPData.get<std::string>("m_base_address");

    if ( sBaseAddress != "not_used" ) {
      ipDataHdr.m_base_address = XUtil::stringToUInt64(sBaseAddress);
    }
    else {
      ipDataHdr.m_base_address = (uint64_t) -1;
    }

    std::string sm_name = ptIPData.get<std::string>("m_name");
    if (sm_name.length() >= sizeof(ip_data::m_name)) {
      std::string errMsg = XUtil::format("ERROR: The m_name entry length (%d), exceeds the allocated space (%d).  Name: '%s'",
                                         (unsigned int)sm_name.length(), (unsigned int)sizeof(ip_data::m_name), sm_name.c_str());
      throw std::runtime_error(errMsg);
    }

    // We already know that there is enough room for this string
    memcpy(ipDataHdr.m_name, sm_name.c_str(), sm_name.length() + 1);

    if ((ipDataHdr.m_type == IP_MEM_DDR4) ||
        (ipDataHdr.m_type == IP_MEM_HBM)) {
      XUtil::TRACE(XUtil::format("[%d]: m_type: %d, m_index: %d, m_pc_index: %d, m_base_address: 0x%lx, m_name: '%s'",
                                 count,
                                 (unsigned int)ipDataHdr.m_type,
                                 (unsigned int)ipDataHdr.indices.m_index,
                                 (unsigned int)ipDataHdr.indices.m_pc_index,
                                 ipDataHdr.m_base_address,
                                 ipDataHdr.m_name));
    } else {
      XUtil::TRACE(XUtil::format("[%d]: m_type: %d, properties: 0x%x, m_base_address: 0x%lx, m_name: '%s'",
                                 count,
                                 (unsigned int)ipDataHdr.m_type,
                                 (unsigned int)ipDataHdr.properties,
                                 ipDataHdr.m_base_address,
                                 ipDataHdr.m_name));
    }

    // Write out the entire structure
    XUtil::TRACE_BUF("ip_data", reinterpret_cast<const char*>(&ipDataHdr), sizeof(ip_data));
    _buf.write(reinterpret_cast<const char*>(&ipDataHdr), sizeof(ip_data));
    count++;
  }

  // -- The counts should match --
  if (count != (unsigned int)ipLayoutHdr.m_count) {
    std::string errMsg = XUtil::format("ERROR: Number of connection sections (%d) does not match expected encoded value: %d",
                                       (unsigned int)count, (unsigned int)ipLayoutHdr.m_count);
    throw std::runtime_error(errMsg);
  }

  // -- Buffer needs to be less than 64K--
  unsigned int bufferSize = (unsigned int) _buf.str().size();
  const unsigned int maxBufferSize = 64 * 1024;
  if ( bufferSize > maxBufferSize ) {
    std::string errMsg = XUtil::format("CRITICAL WARNING: The buffer size for the IP_LAYOUT section (%d) exceed the maximum size of %d.\nThis can result in lose of data in the driver.",
                                       (unsigned int) bufferSize, (unsigned int) maxBufferSize);
    std::cout << errMsg << std::endl;
    // throw std::runtime_error(errMsg);
  }
}


bool 
SectionIPLayout::doesSupportAddFormatType(FormatType _eFormatType) const
{
  if (_eFormatType == FT_JSON) {
    return true;
  }
  return false;
}

bool 
SectionIPLayout::doesSupportDumpFormatType(FormatType _eFormatType) const
{
  if ((_eFormatType == FT_JSON) ||
      (_eFormatType == FT_HTML) ||
      (_eFormatType == FT_RAW))
  {
    return true;
  }

  return false;
}

template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt, 
                         boost::property_tree::ptree::key_type const& key)
{
  std::vector<T> r;
  for (auto& item : pt.get_child(key))
      r.push_back(item.second);
  return r;
}


void 
SectionIPLayout::appendToSectionMetadata(const boost::property_tree::ptree& _ptAppendData,
                                         boost::property_tree::ptree& _ptToAppendTo)
{
  XUtil::TRACE_PrintTree("To Append To", _ptToAppendTo);
  XUtil::TRACE_PrintTree("Append data", _ptAppendData);


  std::vector<boost::property_tree::ptree> ip_datas = as_vector<boost::property_tree::ptree>(_ptAppendData, "m_ip_data");
  unsigned int appendCount = _ptAppendData.get<unsigned int>("m_count");

  if (appendCount != ip_datas.size()) {
    std::string errMsg = XUtil::format("ERROR: IP layout section to append's count (%d) doesn't match the number of ip_data entries (%d).", appendCount, ip_datas.size());
    throw std::runtime_error(errMsg);
  }

  if (appendCount == 0) {
    std::string errMsg = "WARNING: IP layout section doesn't contain any data to append.";
    std::cout << errMsg << std::endl;
    return;
  }

  // Now copy the data
  boost::property_tree::ptree& ptIPLayoutAppendTo = _ptToAppendTo.get_child("ip_layout");
  boost::property_tree::ptree& ptDest_m_ip_data = ptIPLayoutAppendTo.get_child("m_ip_data");

  for (auto ip_data : ip_datas) {
    boost::property_tree::ptree new_ip_data;
    std::string sm_type = ip_data.get<std::string>("m_type");
    new_ip_data.put("m_type", sm_type);

    if ((getIPType(sm_type) == IP_MEM_DDR4) ||
        (getIPType(sm_type) == IP_MEM_HBM)) {
      new_ip_data.put("m_index", ip_data.get<std::string>("m_index"));
      new_ip_data.put("m_pc_index", ip_data.get<std::string>("m_pc_index", "0"));
    } else {
      new_ip_data.put("properties", ip_data.get<std::string>("properties"));
    }
    new_ip_data.put("m_base_address", ip_data.get<std::string>("m_base_address"));
    new_ip_data.put("m_name", ip_data.get<std::string>("m_name"));

    ptDest_m_ip_data.push_back(std::make_pair("", new_ip_data));   // Used to make an array of objects
  }

  // Update count
  {
    unsigned int count = ptIPLayoutAppendTo.get<unsigned int>("m_count");
    count += appendCount;
    ptIPLayoutAppendTo.put("m_count", count);
  }

  XUtil::TRACE_PrintTree("To Append To Done", _ptToAppendTo);
}


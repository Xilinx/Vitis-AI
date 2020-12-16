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

#include "FormattedOutput.h"
#include "Section.h"
#include "SectionBitstream.h"
#include "XclBinSignature.h"
#include  <set>

#include <iostream>
#include <boost/uuid/uuid.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/property_tree/json_parser.hpp>

// Generated include files
#include <version.h>

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt, 
                         boost::property_tree::ptree::key_type const& key)
{
    std::vector<T> r;

    boost::property_tree::ptree::const_assoc_iterator it = pt.find(key);

    if( it != pt.not_found()) {
      for (auto& item : pt.get_child(key)) {
        r.push_back(item.second);
      }
    }
    return r;
}

std::string
FormattedOutput::getTimeStampAsString(const axlf &_xclBinHeader) {
  return XUtil::format("%ld", _xclBinHeader.m_header.m_timeStamp);
}

std::string
FormattedOutput::getFeatureRomTimeStampAsString(const axlf &_xclBinHeader) {
  return XUtil::format("%d", _xclBinHeader.m_header.m_featureRomTimeStamp);
}

std::string
FormattedOutput::getVersionAsString(const axlf &_xclBinHeader) {
  return XUtil::format("%d.%d.%d", _xclBinHeader.m_header.m_versionMajor, _xclBinHeader.m_header.m_versionMinor, _xclBinHeader.m_header.m_versionPatch);
}

// String Getters
std::string 
FormattedOutput::getMagicAsString(const axlf &_xclBinHeader) { 
  return XUtil::format("%s", _xclBinHeader.m_magic); 
}

std::string 
FormattedOutput::getSignatureLengthAsString(const axlf &_xclBinHeader) { 
  std::string sTemp("");
  XUtil::binaryBufferToHexString((unsigned char*)&_xclBinHeader.m_signature_length, sizeof(_xclBinHeader.m_signature_length), sTemp);
  return sTemp; // TBD: "0x" + sTemp; ? do the others too...
}

std::string 
FormattedOutput::getKeyBlockAsString(const axlf &_xclBinHeader) { 
  std::string sTemp("");
  XUtil::binaryBufferToHexString((unsigned char*)&_xclBinHeader.m_keyBlock, sizeof(_xclBinHeader.m_keyBlock), sTemp);
  return sTemp;
}

std::string 
FormattedOutput::getUniqueIdAsString(const axlf &_xclBinHeader) { 
  std::string sTemp("");
  XUtil::binaryBufferToHexString((unsigned char*)&_xclBinHeader.m_uniqueId, sizeof(_xclBinHeader.m_uniqueId), sTemp);
  return sTemp;
}

std::string
getSizeAsString(const axlf &_xclBinHeader) {
  return XUtil::format("%ld", _xclBinHeader.m_header.m_length);
}


std::string
FormattedOutput::getModeAsString(const axlf &_xclBinHeader) {
  return XUtil::format("%d", _xclBinHeader.m_header.m_mode);
}

std::string
getModeAsPrettyString(const axlf &_xclBinHeader) {
  switch (_xclBinHeader.m_header.m_mode) {
    case XCLBIN_FLAT: 
      return "XCLBIN_FLAT";
      break;
    case XCLBIN_PR: 
      return "XCLBIN_PR";
      break;
    case XCLBIN_TANDEM_STAGE2: 
      return "XCLBIN_TANDEM_STAGE2";
      break;
    case XCLBIN_TANDEM_STAGE2_WITH_PR:
      return "XCLBIN_TANDEM_STAGE2_WITH_PR";
      break;
    case XCLBIN_HW_EMU: 
      return "XCLBIN_HW_EMU";
      break;
    case XCLBIN_SW_EMU: 
      return "XCLBIN_SW_EMU";
      break;
    default: 
      return "UNKNOWN";
      break;
  }
}

std::string
FormattedOutput::getFeatureRomUuidAsString(const axlf &_xclBinHeader) {
  std::string sTemp("");
  XUtil::binaryBufferToHexString(_xclBinHeader.m_header.rom_uuid, sizeof(axlf_header::rom_uuid), sTemp);
  return sTemp;
}

std::string
FormattedOutput::getPlatformVbnvAsString(const axlf &_xclBinHeader) {
  return XUtil::format("%s", _xclBinHeader.m_header.m_platformVBNV);
}

std::string
FormattedOutput::getXclBinUuidAsString(const axlf &_xclBinHeader) {
  std::string sTemp("");
  XUtil::binaryBufferToHexString(_xclBinHeader.m_header.uuid, sizeof(axlf_header::uuid), sTemp);
  return sTemp;
}

std::string
FormattedOutput::getDebugBinAsString(const axlf &_xclBinHeader) {
  return XUtil::format("%s", _xclBinHeader.m_header.m_debug_bin);
}


void 
FormattedOutput::getKernelDDRMemory(const std::string _sKernelInstanceName, 
                                    const std::vector<Section*> _sections,
                                    boost::property_tree::ptree &_ptKernelInstance,
                                    boost::property_tree::ptree &_ptMemoryConnections)
{
  if (_sKernelInstanceName.empty()) {
    return;
  }

  // 1) Look for our sections section
  Section *pMemTopology = nullptr;
  Section *pConnectivity = nullptr;
  Section *pIPLayout = nullptr;

  for (auto pSection : _sections) {
    if (MEM_TOPOLOGY == pSection->getSectionKind() ) {
      pMemTopology=pSection; 
    } else if (CONNECTIVITY == pSection->getSectionKind() ) {
      pConnectivity=pSection; 
    } else if (IP_LAYOUT == pSection->getSectionKind() ) {
      pIPLayout=pSection; 
    }
  }
  
  if ((pMemTopology == nullptr) ||
      (pConnectivity == nullptr) ||
      (pIPLayout == nullptr)) {
    // Nothing to work on
    return; 
  }

  // 2) Get the property trees and convert section into vector arrays
  boost::property_tree::ptree ptSections;
  pMemTopology->getPayload(ptSections);
  pConnectivity->getPayload(ptSections);
  pIPLayout->getPayload(ptSections);
  XUtil::TRACE_PrintTree("Top", ptSections);

  boost::property_tree::ptree& ptMemTopology = ptSections.get_child("mem_topology");
  std::vector<boost::property_tree::ptree> memTopology = as_vector<boost::property_tree::ptree>(ptMemTopology, "m_mem_data");

  boost::property_tree::ptree& ptConnectivity = ptSections.get_child("connectivity");
  std::vector<boost::property_tree::ptree> connectivity = as_vector<boost::property_tree::ptree>(ptConnectivity, "m_connection");

  boost::property_tree::ptree& ptIPLayout = ptSections.get_child("ip_layout");
  std::vector<boost::property_tree::ptree> ipLayout = as_vector<boost::property_tree::ptree>(ptIPLayout, "m_ip_data");

  // 3) Establish the connections
  std::set<int> addedIndex;
  for (auto connection : connectivity) {
    unsigned int ipLayoutIndex = connection.get<uint32_t>("m_ip_layout_index");
    unsigned int memDataIndex = connection.get<uint32_t>("mem_data_index");

    if ((_sKernelInstanceName == ipLayout[ipLayoutIndex].get<std::string>("m_name")) &&
        (addedIndex.find(memDataIndex) == addedIndex.end())) {
      _ptMemoryConnections.add_child("mem_data", memTopology[memDataIndex]);
      addedIndex.insert(memDataIndex);
    }
  }

  // 4) Get the kernel information
  for (auto ipdata : ipLayout) {
    if (_sKernelInstanceName == ipdata.get<std::string>("m_name")) {
      _ptKernelInstance.add_child("ip_data", ipdata);
      break;
    }
  }
}

void
reportBuildVersion( std::ostream & _ostream)
{
  _ostream << XUtil::format("%17s: %s", "XRT Build Version", xrt_build_version).c_str() << std::endl;
  _ostream << XUtil::format("%17s: %s", "Build Date", xrt_build_version_date).c_str() << std::endl;
  _ostream << XUtil::format("%17s: %s", "Hash ID", xrt_build_version_hash).c_str() << std::endl;
}

void
FormattedOutput::reportVersion(bool bShort) {
  if (bShort == true) {
    reportBuildVersion( std::cout);
  } else {
    xrt::version::print(std::cout);
  }
}


void
reportXclbinInfo( std::ostream & _ostream,
                  const std::string& _sInputFile,
                  const axlf &_xclBinHeader,
                  boost::property_tree::ptree &_ptMetaData,
                  const std::vector<Section*> _sections)
{
  std::string sSignatureState;

  // Look for the PKCS signature first
  if (!_sInputFile.empty()) {
    try {
      XclBinPKCSImageStats xclBinPKCSStats = {0};
      getXclBinPKCSStats(_sInputFile, xclBinPKCSStats);
  
      if (xclBinPKCSStats.is_PKCS_signed) {
        sSignatureState = XUtil::format("Present - Signed PKCS - Offset: 0x%lx, Size: 0x%lx", xclBinPKCSStats.signature_offset, xclBinPKCSStats.signature_size);
      }
    } catch (...) {
      // Do nothing
    }
  }

  // Calculate if the signature is present or not because this is a slow
  {
    if (!_sInputFile.empty() && sSignatureState.empty()) {
      std::fstream inputStream;
      inputStream.open(_sInputFile, std::ifstream::in | std::ifstream::binary);
      if (inputStream.is_open()) {
        std::string sSignature;
        std::string sSignedBy;
        unsigned int totalSize;
        if (XUtil::getSignature(inputStream, sSignature, sSignedBy, totalSize)) {
          sSignatureState = "Present - " + sSignature;
        }
      }
      inputStream.close();
    }
  }


  _ostream << "xclbin Information" << std::endl;
  _ostream << "------------------" << std::endl;

  // Generated By:
  { 
    std::string sTool = _ptMetaData.get<std::string>("xclbin.generated_by.name","");
    std::string sVersion = _ptMetaData.get<std::string>("xclbin.generated_by.version", "");
    std::string sTimeStamp = _ptMetaData.get<std::string>("xclbin.generated_by.time_stamp", "");
    std::string sGeneratedBy = XUtil::format("%s (%s) on %s", sTool.c_str(), sVersion.c_str(), sTimeStamp.c_str());
    if ( sTool.empty() ) {
      sGeneratedBy = "<unknown>";
    }

    _ostream << XUtil::format("   %-23s %s", "Generated by:", sGeneratedBy.c_str()).c_str() << std::endl;
  }

  // Version:
  _ostream << XUtil::format("   %-23s %d.%d.%d", "Version:", _xclBinHeader.m_header.m_versionMajor, _xclBinHeader.m_header.m_versionMinor, _xclBinHeader.m_header.m_versionPatch).c_str() << std::endl;

  // Kernels
  {
    std::string sKernels;
    if (!_ptMetaData.empty()) {
      boost::property_tree::ptree &ptXclBin = _ptMetaData.get_child("xclbin");
      std::vector<boost::property_tree::ptree> userRegions = as_vector<boost::property_tree::ptree>(ptXclBin,"user_regions");
      for (auto & userRegion : userRegions) {
        std::vector<boost::property_tree::ptree> kernels = as_vector<boost::property_tree::ptree>(userRegion,"kernels");
        for (auto & kernel : kernels) {
           std::string sKernel = kernel.get<std::string>("name", "");
           if (sKernel.empty()) {
             continue;
           }
           if (!sKernels.empty()) {
             sKernels += ", ";
           }
           sKernels += sKernel;
        }
      }
    } else {
      sKernels = "<unknown>";
    }
    _ostream << XUtil::format("   %-23s %s", "Kernels:", sKernels.c_str()).c_str() << std::endl;
  }
 
  // Signature
  {
    _ostream << XUtil::format("   %-23s %s", "Signature:", sSignatureState.c_str()).c_str() << std::endl;
  }

  // Content
  {
    std::string sContent;
    for (auto pSection : _sections) {
      if (pSection->getSectionKind() == BITSTREAM) {
        SectionBitstream *pBitstreamSection = static_cast<SectionBitstream *>(pSection);
        sContent = pBitstreamSection->getContentTypeAsString();
        break;
      }
    }
    _ostream << XUtil::format("   %-23s %s", "Content:", sContent.c_str()).c_str() << std::endl;
  }
   
  {
    std::string sUUID = XUtil::getUUIDAsString(_xclBinHeader.m_header.uuid);
    _ostream << XUtil::format("   %-23s %s", "UUID (xclbin):", sUUID.c_str()).c_str() << std::endl;
  }

  {
    // Get the PARTITION_METADATA property tree (if there is one)
    for (auto pSection : _sections) {
      if (pSection->getSectionKind() != PARTITION_METADATA) {
        continue;
      }

      // Get the complete JSON metadata tree
      boost::property_tree::ptree ptRoot;
      pSection->getPayload(ptRoot);
      if (ptRoot.empty()) {
        continue;
      }

      // Look for the "partition_metadata" node
      boost::property_tree::ptree ptPartitionMetadata = ptRoot.get_child("partition_metadata");
      if (ptPartitionMetadata.empty()) {
        continue;
      }

      // Look for the "interfaces" node
      boost::property_tree::ptree ptInterfaces = ptPartitionMetadata.get_child("interfaces");
      if (ptInterfaces.empty()) {
        continue;
      }

      // Report all of the interfaces
      for (const auto& kv : ptInterfaces) {
        boost::property_tree::ptree ptInterface = kv.second;
        std::string sUUID = ptInterface.get<std::string>("interface_uuid", "");
        if (!sUUID.empty()) {
          _ostream << XUtil::format("   %-23s %s", "UUID (IINTF):", sUUID.c_str()).c_str() << std::endl;
        }
      }
    }
  }

  // Sections
  {
    std::string sSections;
    for (Section *pSection : _sections) {
      std::string sKindStr = pSection->getSectionKindAsString();

      if (!sSections.empty()) {
        sSections += ", ";
      }

      sSections += sKindStr;

      if (!pSection->getSectionIndexName().empty()) {
        sSections += "[" + pSection->getSectionIndexName() + "]";
      }
    }

    std::vector<std::string> sections;
    const unsigned int wrapLength=54;
    std::string::size_type lastPos = 0;
    std::string::size_type pos = 0;
    std::string delimiters = ", ";

    while(pos != std::string::npos)  
    {
       pos = sSections.find(delimiters, lastPos);
       std::string token;

       if (pos == std::string::npos) {
          token = sSections.substr(lastPos, (sSections.length()-lastPos));
       } else {
          token = sSections.substr(lastPos, pos-lastPos+2);
       }

       if (sections.empty()) {
         sections.push_back(token);
       } else {
         unsigned int index = (unsigned int) sections.size() - 1;
         if ((sections[index].length() + token.length()) > wrapLength) {
           sections.push_back(token);
         } else {
           sections[index] += token;
         }
       }

       lastPos = pos + 2;
    }

    for (unsigned index = 0; index < sections.size(); ++index) {
      if (index == 0) {
        _ostream << XUtil::format("   %-23s %s", "Sections:", sections[index].c_str()).c_str() << std::endl;
      } else {
        _ostream << XUtil::format("   %-23s %s", "", sections[index].c_str()).c_str() << std::endl;
      }
    }
  }
}

/*
 * get string value from boost:property_tree, first try platform.<name>
 * then try dsa.<name>.
 */
std::string
getPTreeValue(boost::property_tree::ptree &_ptMetaData, std::string name)
{
  std::string result = _ptMetaData.get<std::string>("platform." + name, "--");
  if (result.compare("--") == 0) {
    result = _ptMetaData.get<std::string>("dsa." + name, "--");
  }
  return result;
}

void
reportHardwarePlatform( std::ostream & _ostream,
                  const axlf &_xclBinHeader,
                  boost::property_tree::ptree &_ptMetaData)
{
  _ostream << "Hardware Platform (Shell) Information" << std::endl;
  _ostream << "-------------------------------------" << std::endl;

  if (!_ptMetaData.empty()) {
    // Vendor
    {
      std::string sVendor = getPTreeValue(_ptMetaData, "vendor");
      _ostream << XUtil::format("   %-23s %s", "Vendor:", sVendor.c_str()).c_str() << std::endl;
    }

    // Board
    {
      std::string sName = getPTreeValue(_ptMetaData, "board_id");
      _ostream << XUtil::format("   %-23s %s", "Board:", sName.c_str()).c_str() << std::endl;
    }

    // Name
    {
      std::string sName = getPTreeValue(_ptMetaData, "name");
      _ostream << XUtil::format("   %-23s %s", "Name:", sName.c_str()).c_str() << std::endl;
    }

    // Version
    {
      std::string sVersion = getPTreeValue(_ptMetaData, "version_major") 
                           + "." 
                           + getPTreeValue(_ptMetaData, "version_minor");
      _ostream << XUtil::format("   %-23s %s", "Version:", sVersion.c_str()).c_str() << std::endl;
    }

    // Generated Version
    {
      std::string sGeneratedVersion = getPTreeValue(_ptMetaData, "generated_by.name") + " "
                                    + getPTreeValue(_ptMetaData, "generated_by.version") 
                                    + " (SW Build: " 
                                    + getPTreeValue(_ptMetaData, "generated_by.cl");
      std::string sIPCL = getPTreeValue(_ptMetaData, "generated_by.ip_cl");
      if (sIPCL.compare("--") != 0) {
        sGeneratedVersion += "; " + sIPCL;
      }
      sGeneratedVersion += ")";

      _ostream << XUtil::format("   %-23s %s", "Generated Version:", sGeneratedVersion.c_str()).c_str() << std::endl;
    }

    // Created
    {
      std::string sCreated = getPTreeValue(_ptMetaData, "generated_by.time_stamp");
      _ostream << XUtil::format("   %-23s %s", "Created:", sCreated.c_str()).c_str() << std::endl;
    }

    // FPGA Device
    {
      std::string sFPGADevice = getPTreeValue(_ptMetaData, "board.part");
      if (sFPGADevice.compare("--") != 0) {
        std::string::size_type pos = sFPGADevice.find("-", 0);

        if (pos == std::string::npos) {
           sFPGADevice = "--";
        } else {
           sFPGADevice = sFPGADevice.substr(0, pos);
        }
      }

      _ostream << XUtil::format("   %-23s %s", "FPGA Device:", sFPGADevice.c_str()).c_str() << std::endl;
    }

    // Board Vendor
    {
       std::string sBoardVendor = getPTreeValue(_ptMetaData, "board.vendor");
      _ostream << XUtil::format("   %-23s %s", "Board Vendor:", sBoardVendor.c_str()).c_str() << std::endl;
    }

    // Board Name
    {
       std::string sBoardName = getPTreeValue(_ptMetaData, "board.name");
      _ostream << XUtil::format("   %-23s %s", "Board Name:", sBoardName.c_str()).c_str() << std::endl;
    }

    // Board Part
    {
       std::string sBoardPart = getPTreeValue(_ptMetaData, "board.board_part");
      _ostream << XUtil::format("   %-23s %s", "Board Part:", sBoardPart.c_str()).c_str() << std::endl;
    }
  }

  // Platform VBNV
  {
    std::string sPlatformVBNV = (char *) _xclBinHeader.m_header.m_platformVBNV;
    if (sPlatformVBNV.empty()) {
      sPlatformVBNV = "<not defined>";
    }
    _ostream << XUtil::format("   %-23s %s", "Platform VBNV:", sPlatformVBNV.c_str()).c_str() << std::endl;
  }

  // Static UUID
  {
    std::string sStaticUUID = XUtil::getUUIDAsString(_xclBinHeader.m_header.rom_uuid);
    _ostream << XUtil::format("   %-23s %s", "Static UUID:", sStaticUUID.c_str()).c_str() << std::endl;
  }

  // TimeStamp
  {
     _ostream << XUtil::format("   %-23s %ld", "Feature ROM TimeStamp:", _xclBinHeader.m_header.m_featureRomTimeStamp).c_str() << std::endl;
  }
 

}


void
reportClocks( std::ostream & _ostream,
              const std::vector<Section*> _sections)
{
  _ostream << "Clocks" << std::endl;
  _ostream << "------" << std::endl;

  boost::property_tree::ptree ptClockFreqTopology;
  for (Section * pSection : _sections) {
    if (pSection->getSectionKind() == CLOCK_FREQ_TOPOLOGY) {
      boost::property_tree::ptree pt;
      pSection->getPayload(pt);
      if (!pt.empty()) {
        ptClockFreqTopology = pt.get_child("clock_freq_topology");
      }
      break;
    }
  }

  if (ptClockFreqTopology.empty()) {
    _ostream << "   No clock frequency data available."  << std::endl;
    return;
  }

  std::vector<boost::property_tree::ptree> clockFreqs = as_vector<boost::property_tree::ptree>(ptClockFreqTopology,"m_clock_freq");
  for (unsigned int index = 0; index < clockFreqs.size(); ++index) {
    boost::property_tree::ptree &ptClockFreq = clockFreqs[index];
    std::string sName = ptClockFreq.get<std::string>("m_name");
    std::string sType = ptClockFreq.get<std::string>("m_type");
    std::string sFreqMhz = ptClockFreq.get<std::string>("m_freq_Mhz");

    _ostream << XUtil::format("   %-10s %s", "Name:", sName.c_str()).c_str() << std::endl;
    _ostream << XUtil::format("   %-10s %d", "Index:", index).c_str() << std::endl;
    _ostream << XUtil::format("   %-10s %s", "Type:", sType.c_str()).c_str() << std::endl;
    _ostream << XUtil::format("   %-10s %s MHz", "Frequency:", sFreqMhz.c_str()).c_str() << std::endl;

    if (&ptClockFreq != &clockFreqs.back()) {
      _ostream << std::endl;
    }
  }
}

void
reportMemoryConfiguration( std::ostream & _ostream,
                           const std::vector<Section*> _sections)
{
  _ostream << "Memory Configuration" << std::endl;
  _ostream << "--------------------" << std::endl;

  boost::property_tree::ptree ptMemTopology;
  for (Section * pSection : _sections) {
    if (pSection->getSectionKind() == MEM_TOPOLOGY) {
      boost::property_tree::ptree pt;
      pSection->getPayload(pt);
      if (!pt.empty()) {
        ptMemTopology = pt.get_child("mem_topology");
      }
      break;
    }
  }

  if (ptMemTopology.empty()) {
    _ostream << "   No memory configuration data available."  << std::endl;
    return;
  }

  std::vector<boost::property_tree::ptree> memDatas = as_vector<boost::property_tree::ptree>(ptMemTopology,"m_mem_data");
  for (unsigned int index = 0; index < memDatas.size(); ++index) {
    boost::property_tree::ptree & ptMemData = memDatas[index];

    std::string sName = ptMemData.get<std::string>("m_tag");
    std::string sType = ptMemData.get<std::string>("m_type");
    std::string sBaseAddress = ptMemData.get<std::string>("m_base_address");
    std::string sAddressSizeKB = ptMemData.get<std::string>("m_sizeKB");
    uint64_t addressSize = XUtil::stringToUInt64(sAddressSizeKB) * 1024;
    std::string sUsed = ptMemData.get<std::string>("m_used");
  
    std::string sBankUsed = "No";

    if (sUsed != "0") {
      sBankUsed = "Yes";
    }

    _ostream << XUtil::format("   %-13s %s", "Name:", sName.c_str()).c_str() << std::endl;
    _ostream << XUtil::format("   %-13s %d", "Index:", index).c_str() << std::endl;
    _ostream << XUtil::format("   %-13s %s", "Type:", sType.c_str()).c_str() << std::endl;
    _ostream << XUtil::format("   %-13s %s", "Base Address:", sBaseAddress.c_str()).c_str() << std::endl;
    _ostream << XUtil::format("   %-13s 0x%lx", "Address Size:", addressSize).c_str() << std::endl;
    _ostream << XUtil::format("   %-13s %s", "Bank Used:", sBankUsed.c_str()).c_str() << std::endl;

    if (&ptMemData != &memDatas.back()) {
      _ostream << std::endl;
    }
  }
}

void
reportKernels( std::ostream & _ostream,
               boost::property_tree::ptree &_ptMetaData,
               const std::vector<Section*> _sections)
{
  if (_ptMetaData.empty()) {
    _ostream << "   No kernel metadata available."  << std::endl;
    return;
  }

  // Cross reference data
  std::vector<boost::property_tree::ptree> memTopology;
  std::vector<boost::property_tree::ptree> connectivity;
  std::vector<boost::property_tree::ptree> ipLayout;

  for (auto pSection : _sections) {
    boost::property_tree::ptree pt;
    if (MEM_TOPOLOGY == pSection->getSectionKind() ) {
      pSection->getPayload(pt);
      memTopology = as_vector<boost::property_tree::ptree>(pt.get_child("mem_topology"), "m_mem_data");
    } else if (CONNECTIVITY == pSection->getSectionKind() ) {
      pSection->getPayload(pt);
      connectivity = as_vector<boost::property_tree::ptree>(pt.get_child("connectivity"), "m_connection");
    } else if (IP_LAYOUT == pSection->getSectionKind() ) {
      pSection->getPayload(pt);
      ipLayout = as_vector<boost::property_tree::ptree>(pt.get_child("ip_layout"), "m_ip_data");
    }
  }

  // All or nothing
  if (!(memTopology.empty() && connectivity.empty() && ipLayout.empty())) {
    if (memTopology.empty()) {
      std::string errMsg = "ERROR: Missing MEM_TOPOLOGY section.  This usually is an indication of a malformed xclbin archive.";
      throw std::runtime_error(errMsg);
    }

//    if (connectivity.empty()) {
//      std::string errMsg = "ERROR: Missing CONNECTIVITY section.  This usually is an indication of a malformed xclbin archive.";
//      throw std::runtime_error(errMsg);
//    }

    if (ipLayout.empty()) {
      std::string errMsg = "ERROR: Missing IP_LAYOUT section.  This usually is an indication of a malformed xclbin archive.";
      throw std::runtime_error(errMsg);
    }
  }

  boost::property_tree::ptree &ptXclBin = _ptMetaData.get_child("xclbin");
  std::vector<boost::property_tree::ptree> userRegions = as_vector<boost::property_tree::ptree>(ptXclBin,"user_regions");
  for (auto & userRegion : userRegions) {
    std::vector<boost::property_tree::ptree> kernels = as_vector<boost::property_tree::ptree>(userRegion,"kernels");
    for (auto & ptKernel : kernels) {
      XUtil::TRACE_PrintTree("Kernel", ptKernel);

      std::string sKernel = ptKernel.get<std::string>("name");
      _ostream << XUtil::format("%s %s", "Kernel:", sKernel.c_str()).c_str() << std::endl;

      std::vector<boost::property_tree::ptree> ports = as_vector<boost::property_tree::ptree>(ptKernel,"ports");
      std::vector<boost::property_tree::ptree> arguments = as_vector<boost::property_tree::ptree>(ptKernel,"arguments");
      std::vector<boost::property_tree::ptree> instances = as_vector<boost::property_tree::ptree>(ptKernel,"instances");

      _ostream << std::endl;

      // Definition
      {
        _ostream << "Definition" << std::endl;
        _ostream << "----------" << std::endl;

        _ostream << "   Signature: " << sKernel << " (";
        for (auto & ptArgument : arguments) {
          std::string sType = ptArgument.get<std::string>("type");
          std::string sName = ptArgument.get<std::string>("name");

          _ostream << sType << " " << sName;
          if (&ptArgument != &arguments.back())
            _ostream << ", ";
        }
        _ostream << ")" << std::endl;
      }

      _ostream << std::endl;

      // Ports
      {
        _ostream << "Ports" << std::endl;
        _ostream << "-----" << std::endl;

        for (auto & ptPort : ports) {
          std::string sPort = ptPort.get<std::string>("name");
          std::string sMode = ptPort.get<std::string>("mode");
          std::string sRangeBytes = ptPort.get<std::string>("range");
          std::string sDataWidthBits = ptPort.get<std::string>("data_width");
          std::string sPortType = ptPort.get<std::string>("port_type");

          _ostream << XUtil::format("   %-14s %s", "Port:", sPort.c_str()).c_str() << std::endl;
          _ostream << XUtil::format("   %-14s %s", "Mode:", sMode.c_str()).c_str() << std::endl;
          _ostream << XUtil::format("   %-14s %s", "Range (bytes):", sRangeBytes.c_str()).c_str() << std::endl;
          _ostream << XUtil::format("   %-14s %s bits", "Data Width:", sDataWidthBits.c_str()).c_str() << std::endl;
          _ostream << XUtil::format("   %-14s %s", "Port Type:", sPortType.c_str()).c_str() << std::endl;

          if (&ptPort != &ports.back()) {
            _ostream << std::endl;
          }
        }
      }

      _ostream << std::endl;

      // Instance
      for (auto & ptInstance : instances) {
        _ostream << "--------------------------" << std::endl;
        std::string sInstance = ptInstance.get<std::string>("name");
        _ostream << XUtil::format("%-16s %s", "Instance:", sInstance.c_str()).c_str() << std::endl;

        std::string sKernelInstance = sKernel + ":" + sInstance;

        // Base Address
        {
          std::string sBaseAddress = "--";
          for (auto & ptIPData : ipLayout ) {
            if (ptIPData.get<std::string>("m_name") == sKernelInstance) {
              sBaseAddress = ptIPData.get<std::string>("m_base_address");
              break;
            }
          }
          _ostream << XUtil::format("   %-13s %s", "Base Address:", sBaseAddress.c_str()).c_str() << std::endl;
        }

        _ostream << std::endl;

        // List the arguments
        for (unsigned int argumentIndex = 0; argumentIndex < arguments.size(); ++argumentIndex) {
          boost::property_tree::ptree &ptArgument = arguments[argumentIndex];
          std::string sArgument = ptArgument.get<std::string>("name");
          std::string sOffset = ptArgument.get<std::string>("offset");
          std::string sPort = ptArgument.get<std::string>("port");

          _ostream << XUtil::format("   %-18s %s", "Argument:", sArgument.c_str()).c_str() << std::endl;
          _ostream << XUtil::format("   %-18s %s", "Register Offset:", sOffset.c_str()).c_str() << std::endl;
          _ostream << XUtil::format("   %-18s %s", "Port:", sPort.c_str()).c_str() << std::endl;

          // Find the memory connections
          bool bFoundMemConnection = false;
          for (auto &ptConnection : connectivity) {
            unsigned int ipIndex = ptConnection.get<unsigned int>("m_ip_layout_index");

            if (ipIndex >= ipLayout.size()) {
              std::string errMsg = XUtil::format("ERROR: connectivity section 'm_ip_layout_index' (%d) exceeds the number of 'ip_layout' elements (%d).  This is usually an indication of curruptions in the xclbin archive.", ipIndex, ipLayout.size());
              throw std::runtime_error(errMsg);
            }

            if (ipLayout[ipIndex].get<std::string>("m_name") == sKernelInstance) {
              if (ptConnection.get<unsigned int>("arg_index") == argumentIndex) {
                bFoundMemConnection = true;

                unsigned int memIndex = ptConnection.get<unsigned int>("mem_data_index");
                if (memIndex >= memTopology.size()) {
                  std::string errMsg = XUtil::format("ERROR: connectivity section 'mem_data_index' (%d) exceeds the number of 'mem_topology' elements (%d).  This is usually an indication of curruptions in the xclbin archive.", memIndex, memTopology.size());
                  throw std::runtime_error(errMsg);
                }

                std::string sMemName = memTopology[memIndex].get<std::string>("m_tag");
                std::string sMemType = memTopology[memIndex].get<std::string>("m_type");

                _ostream << XUtil::format("   %-18s %s (%s)", "Memory:", sMemName.c_str(), sMemType.c_str()).c_str() << std::endl;
              }
            }
          }
          if (!bFoundMemConnection) {
            _ostream << XUtil::format("   %-18s <not applicable>", "Memory:").c_str() << std::endl;
          }
          if (argumentIndex != (arguments.size()-1)) {
            _ostream << std::endl;
          }
        }
        if (&ptInstance != &instances.back()) {
          _ostream << std::endl;
        }
      }
    }
  }
}

void
reportXOCC( std::ostream & _ostream,
            boost::property_tree::ptree &_ptMetaData)
{
  if (_ptMetaData.empty()) {
    _ostream << "   No information regarding the creation of the xclbin acceleration image."  << std::endl;
    return;
  }

  _ostream << "Generated By" << std::endl;
  _ostream << "------------" << std::endl;

  // Command
  std::string sCommand = _ptMetaData.get<std::string>("xclbin.generated_by.name","");

  if (sCommand.empty()) {
    _ostream << "   < Data not available >" << std::endl;
    return;
  }

  _ostream << XUtil::format("   %-14s %s", "Command:", sCommand.c_str()).c_str() << std::endl;


  // Version
  {
    std::string sVersion = _ptMetaData.get<std::string>("xclbin.generated_by.version","--");
    std::string sCL = _ptMetaData.get<std::string>("xclbin.generated_by.cl","--");
    std::string sTimeStamp = _ptMetaData.get<std::string>("xclbin.generated_by.time_stamp","--");

    _ostream << XUtil::format("   %-14s %s - %s (SW BUILD: %s)", "Version:", sVersion.c_str(), sTimeStamp.c_str(), sCL.c_str()).c_str() << std::endl;
  }

  std::string sCommandLine = _ptMetaData.get<std::string>("xclbin.generated_by.options","");

  // Command Line
  {
    std::string::size_type pos = sCommandLine.find(" ", 0);
    std::string sOptions;
    if (pos == std::string::npos) {
      sOptions = sCommandLine;
    } else {
      sOptions = sCommandLine.substr(pos+1);
    }

    _ostream << XUtil::format("   %-14s %s %s", "Command Line:", sCommand.c_str(), sOptions.c_str()).c_str() << std::endl;
  }
  
  // Options  
  {
    const std::string delimiters = " -";      // Our delimiter

    // Working variables
    std::string::size_type pos = 0;
    std::string::size_type lastPos = 0;
    std::vector<std::string> commandAndOptions;

    // Parse the string until the entire string has been parsed or 3 tokens have been found
    while(true)  
    {
       pos = sCommandLine.find(delimiters, lastPos);
       std::string token;

       if (pos == std::string::npos) {
          pos = sCommandLine.length();
          commandAndOptions.push_back(sCommandLine.substr(lastPos, pos-lastPos));
          break;
       }

       commandAndOptions.push_back(sCommandLine.substr(lastPos, pos-lastPos));
       lastPos = ++pos ;
    }

    for (unsigned int index = 1; index < commandAndOptions.size(); ++index) {
      if (index == 1) {
        _ostream << XUtil::format("   %-14s %s", "Options:", commandAndOptions[index].c_str()).c_str() << std::endl;
      } else {
        _ostream << XUtil::format("   %-14s %s", "", commandAndOptions[index].c_str()).c_str() << std::endl;
      }
    }
  }
}

void
reportKeyValuePairs( std::ostream & _ostream,
                     const std::vector<Section*> _sections)
{
  _ostream << "User Added Key Value Pairs" << std::endl;
  _ostream << "--------------------------" << std::endl;

 std::vector<boost::property_tree::ptree> keyValues;

  for (Section *pSection : _sections) {
    if (pSection->getSectionKind() == KEYVALUE_METADATA) {
      boost::property_tree::ptree pt;
      pSection->getPayload(pt);
      keyValues = as_vector<boost::property_tree::ptree>(pt.get_child("keyvalue_metadata"), "key_values");
      break;
    }
  }

  if (keyValues.empty()) {
    _ostream << "   <empty>" << std::endl;
    return;
  }

  for (unsigned int index = 0; index < keyValues.size(); ++index) {
    std::string sKey = keyValues[index].get<std::string>("key");
    std::string sValue = keyValues[index].get<std::string>("value");
    _ostream << XUtil::format("   %d) '%s' = '%s'", index+1, sKey.c_str(), sValue.c_str()).c_str() << std::endl;
  }
}

void
reportAllJsonMetadata( std::ostream & _ostream,
                      const std::vector<Section*> _sections)
{
  _ostream << "JSON Metadata for Supported Sections" << std::endl;
  _ostream << "------------------------------------" << std::endl;

  boost::property_tree::ptree pt;
  for (Section * pSection : _sections) {
    std::string sectionName = pSection->getSectionKindAsString();
    XUtil::TRACE("Examining: '" + sectionName);
    pSection->getPayload(pt);
  }

  boost::property_tree::write_json(_ostream, pt, true /*Pretty print*/);
}


void
FormattedOutput::reportInfo(std::ostream &_ostream, 
                            const std::string& _sInputFile,
                            const axlf &_xclBinHeader, 
                            const std::vector<Section*> _sections,
                            bool _bVerbose) {
  // Get the Metadata
  boost::property_tree::ptree ptMetaData;

  for ( Section *pSection : _sections) {
    if (pSection->getSectionKind() == BUILD_METADATA) {
      boost::property_tree::ptree pt;
      pSection->getPayload( pt );
      ptMetaData = pt.get_child("build_metadata", pt);
      break;
    }
  }

  _ostream << std::endl << std::string(78,'=') << std::endl;

  reportBuildVersion(_ostream);
  _ostream << std::string(78,'=') << std::endl;

  if (ptMetaData.empty()) {
    _ostream << "The BUILD_METADATA section is not present. Reports will be limited." << std::endl;
    _ostream << std::string(78,'=') << std::endl;
  }

  reportXclbinInfo(_ostream, _sInputFile, _xclBinHeader, ptMetaData, _sections);
  _ostream << std::string(78,'=') << std::endl;

  reportHardwarePlatform(_ostream, _xclBinHeader, ptMetaData);
  _ostream << std::endl;

  reportClocks(_ostream, _sections);
  _ostream << std::endl;

  reportMemoryConfiguration(_ostream, _sections);
  _ostream << std::string(78,'=') << std::endl;

  if (!ptMetaData.empty()) {
    reportKernels(_ostream, ptMetaData, _sections);
    _ostream << std::string(78,'=') << std::endl;

    reportXOCC(_ostream, ptMetaData);
    _ostream << std::string(78,'=') << std::endl;
  }

  reportKeyValuePairs(_ostream, _sections);
  _ostream << std::string(78,'=') << std::endl;

  if (_bVerbose) {
    reportAllJsonMetadata(_ostream, _sections);
    _ostream << std::string(78,'=') << std::endl;
  }

}
  


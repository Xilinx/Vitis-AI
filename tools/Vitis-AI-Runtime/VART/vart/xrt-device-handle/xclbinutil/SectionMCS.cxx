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

#include "SectionMCS.h"

#include "XclBinUtilities.h"
#include <boost/algorithm/string.hpp>

// Disable windows compiler warnings
#ifdef _WIN32
  #pragma warning( disable : 4100)      // 4100 - Unreferenced formal parameter
#endif


namespace XUtil = XclBinUtilities;

// Static Variables / Classes
SectionMCS::_init SectionMCS::_initializer;

// --------------------------------------------------------------------------

SectionMCS::SectionMCS() {
  // Empty
}

// --------------------------------------------------------------------------

SectionMCS::~SectionMCS() {
  // Empty
}

// --------------------------------------------------------------------------

void
SectionMCS::marshalToJSON(char* _pDataSegment,
                          unsigned int _segmentSize,
                          boost::property_tree::ptree& _ptree) const {
  XUtil::TRACE("");
  XUtil::TRACE("Extracting: MCS");

  // Do we have enough room to overlay the header structure
  if (_segmentSize < sizeof(mcs)) {
    throw std::runtime_error(XUtil::format("ERROR: Segment size (%d) is smaller than the size of the mcs structure (%d)",
                                           _segmentSize, sizeof(mcs)));
  }

  mcs* pHdr = (mcs*)_pDataSegment;

  boost::property_tree::ptree pt_mcs;

  XUtil::TRACE(XUtil::format("m_count: %d", (uint32_t)pHdr->m_count));
  XUtil::TRACE_BUF("mcs", reinterpret_cast<const char*>(pHdr), ((uint64_t)&(pHdr->m_chunk[0]) - (uint64_t)pHdr));

  // Do we have something to extract.  Note: This should never happen.
  if (pHdr->m_count == 0) {
    XUtil::TRACE("m_count is zero, nothing to extract");
    return;
  }

  pt_mcs.put("count", XUtil::format("%d", (unsigned int)pHdr->m_count).c_str());

  // Check to make sure that the array did not exceed its bounds
  uint64_t arraySize = ((uint64_t)&(pHdr->m_chunk[0]) - (uint64_t)pHdr) + (sizeof(mcs_chunk) * pHdr->m_count);

  if (arraySize > _segmentSize) {
    throw std::runtime_error(XUtil::format("ERROR: m_chunk array size (0x%lx) exceeds segment size (0x%lx).",
                                           arraySize, _segmentSize));
  }

  // Examine and extract the data
  for (int index = 0; index < pHdr->m_count; ++index) {
    boost::property_tree::ptree pt_mcs_chunk;
    XUtil::TRACE(XUtil::format("[%d]: m_type: %s, m_offset: 0x%lx, m_size: 0x%lx",
                               index,
                               getMCSTypeStr((enum MCS_TYPE)pHdr->m_chunk[index].m_type).c_str(),
                               pHdr->m_chunk[index].m_offset,
                               pHdr->m_chunk[index].m_size));

    XUtil::TRACE_BUF("m_chunk", reinterpret_cast<const char*>(&(pHdr->m_chunk[index])), sizeof(mcs_chunk));

    // Do some error checking
    char* ptrImageBase = _pDataSegment + pHdr->m_chunk[index].m_offset;

    // Check to make sure that the MCS image is partially looking good
    if ((uint64_t)ptrImageBase > ((uint64_t)_pDataSegment) + _segmentSize) {
      throw std::runtime_error(XUtil::format("ERROR: MCS image %d start offset exceeds MCS segment size.", index));
    }

    if (((uint64_t)ptrImageBase) + pHdr->m_chunk[index].m_size > ((uint64_t)_pDataSegment) + _segmentSize) {
      throw std::runtime_error(XUtil::format("ERROR: MCS image %d size exceeds the MCS segment size.", index));
    }

    pt_mcs_chunk.put("m_type", getMCSTypeStr((enum MCS_TYPE)pHdr->m_chunk[index].m_type).c_str());
    pt_mcs_chunk.put("m_offset", XUtil::format("0x%ld", pHdr->m_chunk[index].m_offset).c_str());
    pt_mcs_chunk.put("m_size", XUtil::format("0x%ld", pHdr->m_chunk[index].m_size).c_str());
  }

  // TODO: Add support to write out this data
}

// --------------------------------------------------------------------------

bool
SectionMCS::supportsSubSection(const std::string& _sSubSectionName) const {
  if (getMCSTypeEnum(_sSubSectionName) != MCS_UNKNOWN) {
    return true;
  }

  return false;
}

// --------------------------------------------------------------------------

void
SectionMCS::getSubPayload(char* _pDataSection,
                          unsigned int _sectionSize,
                          std::ostringstream& _buf,
                          const std::string& _sSubSectionName,
                          enum Section::FormatType _eFormatType) const {
  // Make sure we support the subsystem
  if (supportsSubSection(_sSubSectionName) == false) {
    std::string errMsg = XUtil::format("ERROR: For section '%s' the subsystem '%s' is not supported.", getSectionKindAsString().c_str(), _sSubSectionName.c_str());
    throw std::runtime_error(errMsg);
  }

  // Make sure we support the format type
  if (_eFormatType != FT_RAW) {
    std::string errMsg = XUtil::format("ERROR: For section '%s' the format type (%d) is not supported.", getSectionKindAsString().c_str(), _eFormatType);
    throw std::runtime_error(errMsg);
  }

  // Get the payload
  std::vector<mcsBufferPair> mcsBuffers;

  if (m_pBuffer != nullptr) {
    extractBuffers(m_pBuffer, m_bufferSize, mcsBuffers);
  }

  enum MCS_TYPE eMCSType = getMCSTypeEnum(_sSubSectionName);

  for (auto mcsBuffer : mcsBuffers) {
    if (mcsBuffer.first == eMCSType) {
      const std::string &sBuffer = mcsBuffer.second->str();
      _buf.write(sBuffer.c_str(), sBuffer.size());
    }
  }
}

// --------------------------------------------------------------------------

void
SectionMCS::extractBuffers(const char* _pDataSection,
                           unsigned int _sectionSize,
                           std::vector<mcsBufferPair>& _mcsBuffers) const {
  XUtil::TRACE("Extracting: MCS buffers");

  // Do we have enough room to overlay the header structure
  if (_sectionSize < sizeof(mcs)) {
    std::string errMsg = XUtil::format("ERROR: Section size (%d) is smaller than the size of the mcs structure (%d)", _sectionSize, sizeof(mcs));
    throw std::runtime_error(errMsg);
  }

  mcs* pHdr = (mcs*)_pDataSection;

  XUtil::TRACE(XUtil::format("m_count: %d", (uint32_t)pHdr->m_count));
  XUtil::TRACE_BUF("mcs", reinterpret_cast<const char*>(pHdr), ((uint64_t)&(pHdr->m_chunk[0]) - (uint64_t)pHdr));

  // Do we have something to extract.  Note: This should never happen.
  if (pHdr->m_count == 0) {
    XUtil::TRACE("m_count is zero, nothing to extract");
    return;
  }

  // Check to make sure that the array did not exceed its bounds
  uint64_t arraySize = ((uint64_t)&(pHdr->m_chunk[0]) - (uint64_t)pHdr) + (sizeof(mcs_chunk) * pHdr->m_count);

  if (arraySize > _sectionSize) {
    std::string errMsg = XUtil::format("ERROR: m_chunk array size (0x%lx) exceeds segment size (0x%lx).", arraySize, _sectionSize);
    throw std::runtime_error(errMsg);
  }

  // Examine and extract the data
  for (int index = 0; index < pHdr->m_count; ++index) {
    XUtil::TRACE(XUtil::format("[%d]: m_type: %s, m_offset: 0x%lx, m_size: 0x%lx",
                               index,
                               getMCSTypeStr((enum MCS_TYPE)pHdr->m_chunk[index].m_type).c_str(),
                               pHdr->m_chunk[index].m_offset,
                               pHdr->m_chunk[index].m_size));

    XUtil::TRACE_BUF("m_chunk", reinterpret_cast<const char*>(&(pHdr->m_chunk[index])), sizeof(mcs_chunk));

    const char* ptrImageBase = _pDataSection + pHdr->m_chunk[index].m_offset;

    // Check to make sure that the MCS image is partially looking good
    if ((uint64_t)ptrImageBase > ((uint64_t)_pDataSection) + _sectionSize) {
      std::string errMsg = XUtil::format("ERROR: MCS image %d start offset exceeds MCS segment size.", index);
      throw std::runtime_error(errMsg);
    }

    if (((uint64_t)ptrImageBase) + pHdr->m_chunk[index].m_size > ((uint64_t)_pDataSection) + _sectionSize) {
      std::string errMsg = XUtil::format("ERROR: MCS image %d size exceeds the MCS segment size.", index);
      throw std::runtime_error(errMsg);
    }

    std::ostringstream* pBuffer = new std::ostringstream;
    pBuffer->write(ptrImageBase, pHdr->m_chunk[index].m_size);

    _mcsBuffers.emplace_back((enum MCS_TYPE)pHdr->m_chunk[index].m_type, pBuffer);
  }
}

// --------------------------------------------------------------------------

void
SectionMCS::buildBuffer(const std::vector<mcsBufferPair>& _mcsBuffers,
                        std::ostringstream& _buffer) const {
  XUtil::TRACE("Building: MCS buffers");

  // Must have something to work with
  int count = (int) _mcsBuffers.size();
  if (count == 0)
    return;

  mcs mcsHdr = mcs {0};
  mcsHdr.m_count = (int8_t)count;

  XUtil::TRACE(XUtil::format("m_count: %d", (int)mcsHdr.m_count).c_str());

  // Write out the entire structure except for the mcs structure
  XUtil::TRACE_BUF("mcs - minus mcs_chunk", reinterpret_cast<const char*>(&mcsHdr), (sizeof(mcs) - sizeof(mcs_chunk)));
  _buffer.write(reinterpret_cast<const char*>(&mcsHdr), (sizeof(mcs) - sizeof(mcs_chunk)));


  // Calculate The mcs_chunks data
  std::vector<mcs_chunk> mcsChunks;
  {
    uint64_t currentOffset = ((sizeof(mcs) - sizeof(mcs_chunk)) +
                              (sizeof(mcs_chunk) * count));

    for (auto mcsEntry : _mcsBuffers) {
      mcs_chunk mcsChunk = mcs_chunk {0};
      mcsChunk.m_type = (uint8_t) mcsEntry.first;   // Record the MCS type

      mcsEntry.second->seekp(0, std::ios_base::end);
      mcsChunk.m_size = mcsEntry.second->tellp();
      mcsChunk.m_offset = currentOffset;
      currentOffset += mcsChunk.m_size;

      mcsChunks.push_back(mcsChunk);
    }
  }

  // Finish building the buffer
  // First the array
  {
    int index = 0;
    for (auto mcsChunk : mcsChunks) {
      XUtil::TRACE(XUtil::format("[%d]: m_type: %d, m_offset: 0x%lx, m_size: 0x%lx",
                                 index++,
                                 mcsChunk.m_type,
                                 mcsChunk.m_offset,
                                 mcsChunk.m_size));
      XUtil::TRACE_BUF("mcs_chunk", reinterpret_cast<const char*>(&mcsChunk), sizeof(mcs_chunk));
      _buffer.write(reinterpret_cast<const char*>(&mcsChunk), sizeof(mcs_chunk));
    }
  }

  // Second the data
  {
    for (auto mcsEntry : _mcsBuffers) {
      const std::string& stringBuffer = mcsEntry.second->str();
      _buffer.write(stringBuffer.c_str(), stringBuffer.size());
    }
  }
}

// --------------------------------------------------------------------------

void
SectionMCS::readSubPayload(const char* _pOrigDataSection,
                           unsigned int _origSectionSize,
                           std::fstream& _istream,
                           const std::string& _sSubSection,
                           enum Section::FormatType _eFormatType,
                           std::ostringstream& _buffer) const {
  // Determine subsection name
  enum MCS_TYPE eMCSType = getMCSTypeEnum(_sSubSection);

  if (eMCSType == MCS_UNKNOWN) {
    std::string errMsg = XUtil::format("ERROR: Not support subsection '%s' for section '%s',", _sSubSection.c_str(), getSectionKindAsString().c_str());
    throw std::runtime_error(errMsg);
  }

  // Validate format type
  if (_eFormatType != Section::FT_RAW) {
      std::string errMsg = XUtil::format("ERROR: Section '%s' only supports 'RAW' subsections.", getSectionKindAsString().c_str());
    throw std::runtime_error(errMsg);
  }

  // Get any previous sections
  std::vector<mcsBufferPair> mcsBuffers;

  if (_pOrigDataSection != nullptr) {
    extractBuffers(_pOrigDataSection, _origSectionSize, mcsBuffers);
  }

  // Check to see if subsection already exists
  for (auto mcsEntry : mcsBuffers) {
    if (mcsEntry.first == eMCSType) {
      std::string errMsg = XUtil::format("ERROR: Subsection '%s' already exists for section '%s',", _sSubSection.c_str(), getSectionKindAsString().c_str());
      throw std::runtime_error(errMsg);
    }
  }

  // Things are good, now get this new buffer
  {
    _istream.seekg(0, _istream.end);
    uint64_t mcsSize = _istream.tellg();

    // -- Read contents into memory buffer --
    std::unique_ptr<unsigned char> memBuffer(new unsigned char[mcsSize]);
    _istream.clear();
    _istream.seekg(0, _istream.beg);
    _istream.read((char*)memBuffer.get(), mcsSize);

    std::ostringstream* buffer = new std::ostringstream;
    buffer->write(reinterpret_cast<const char*>(memBuffer.get()), mcsSize);
    mcsBuffers.emplace_back(eMCSType, buffer);
  }

  // Now create a new buffer stream
  buildBuffer(mcsBuffers, _buffer);

  // Clean up the memory
  for (auto mcsEntry : mcsBuffers) {
    delete mcsEntry.second;
    mcsEntry.second = nullptr;
  }
}


// --------------------------------------------------------------------------

bool
SectionMCS::subSectionExists(const std::string& _sSubSectionName) const {
  // Get a list of the sections
  std::vector<mcsBufferPair> mcsBuffers;
  if (m_pBuffer != nullptr) {
    extractBuffers(m_pBuffer, m_bufferSize, mcsBuffers);
  }

  // Search for the given section
  enum MCS_TYPE eMCSType = getMCSTypeEnum(_sSubSectionName);
  for (auto mcsBuffer : mcsBuffers) {
    if (mcsBuffer.first == eMCSType) {
      return true;
    }
  }

  // If we get here, then the section of interest doesn't exist
  return false;
}

// --------------------------------------------------------------------------

const std::string
SectionMCS::getMCSTypeStr(enum MCS_TYPE _mcsType) const {
  switch (_mcsType) {
    case MCS_PRIMARY:
      return "MCS_PRIMARY";
    case MCS_SECONDARY:
      return "MCS_SECONDARY";
    case MCS_UNKNOWN:
    default:
      return XUtil::format("UNKNOWN (%d)", (unsigned int)_mcsType);
  }
}

// --------------------------------------------------------------------------

enum MCS_TYPE
SectionMCS::getMCSTypeEnum(const std::string& _sSubSectionType) const {
  // Case-insensitive
  std::string sSubSection = _sSubSectionType;
  boost::to_upper(sSubSection);

  enum MCS_TYPE eMCSType = MCS_UNKNOWN;
  if (sSubSection == "PRIMARY") {eMCSType = MCS_PRIMARY;} 
  if (sSubSection == "SECONDARY") {eMCSType = MCS_SECONDARY;}

  return eMCSType;
}

// --------------------------------------------------------------------------

void 
SectionMCS::writeSubPayload(const std::string & _sSubSectionName,
                            FormatType _eFormatType, 
                            std::fstream&  _oStream) const {
  // Validate format type
  if (_eFormatType != Section::FT_RAW) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' only supports 'RAW' subsections.", getSectionKindAsString().c_str());
    throw std::runtime_error(errMsg);
  }

  // Obtain the collection of MCS buffers
  std::vector<mcsBufferPair> mcsBuffers;
  if (m_pBuffer != nullptr) {
    extractBuffers(m_pBuffer, m_bufferSize, mcsBuffers);
  }

  // Search for the collection of interest
  enum MCS_TYPE eMCSType = getMCSTypeEnum(_sSubSectionName);
  for (auto mcsBuffer : mcsBuffers) {
    if (mcsBuffer.first == eMCSType) {
      const std::string &buffer = mcsBuffer.second->str();
      _oStream.write(buffer.c_str(), buffer.size());
      return;
    }
  }

  // No collection entry
  std::string errMsg = XUtil::format("ERROR: Subsection '%s' of section '%s' does not exist", _sSubSectionName.c_str(), getSectionKindAsString().c_str());
  throw std::runtime_error(errMsg);
}



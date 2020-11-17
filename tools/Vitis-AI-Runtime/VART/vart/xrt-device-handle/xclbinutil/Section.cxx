/**
 * Copyright (C) 2018 - 2019 Xilinx, Inc
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

#include "Section.h"

#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>


#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;


// Disable windows compiler warnings
#ifdef _WIN32
  #pragma warning( disable : 4100)      // 4100 - Unreferenced formal parameter
#endif



// Static Variables Initialization
std::map<enum axlf_section_kind, std::string> Section::m_mapIdToName;
std::map<std::string, enum axlf_section_kind> Section::m_mapNameToId;
std::map<enum axlf_section_kind, Section::Section_factory> Section::m_mapIdToCtor;
std::map<std::string, enum axlf_section_kind> Section::m_mapJSONNameToKind;
std::map<enum axlf_section_kind, bool> Section::m_mapIdToSubSectionSupport;
std::map<enum axlf_section_kind, bool> Section::m_mapIdToSectionIndexSupport;

Section::Section()
    : m_eKind(BITSTREAM)
    , m_sKindName("")
    , m_sIndexName("")
    , m_pBuffer(nullptr)
    , m_bufferSize(0)
    , m_name("") {
  // Empty
}

Section::~Section() {
  purgeBuffers();
}

void
Section::purgeBuffers()
{
  if (m_pBuffer != nullptr) {
    delete m_pBuffer;
    m_pBuffer = nullptr;
  }
  m_bufferSize = 0;
}

void
Section::setName(const std::string &_sSectionName)
{
   m_name = _sSectionName;
}

void
Section::getKinds(std::vector< std::string > & kinds) {
  for (auto & item : m_mapNameToId) {
    kinds.push_back(item.first);
  }
}

void
Section::registerSectionCtor(enum axlf_section_kind _eKind,
                             const std::string& _sKindStr,
                             const std::string& _sHeaderJSONName,
                             bool _bSupportsSubSections,
                             bool _bSupportsIndexing,
                             Section_factory _Section_factory) {
  // Some error checking
  if (_sKindStr.empty()) {
    std::string errMsg = XUtil::format("ERROR: Kind (%d) pretty print name is missing.", _eKind);
    throw std::runtime_error(errMsg);
  }

  if (m_mapIdToName.find(_eKind) != m_mapIdToName.end()) {
    std::string errMsg = XUtil::format("ERROR: Attempting to register (%d : %s). Constructor enum of kind (%d) already registered.",
                                       (unsigned int)_eKind, _sKindStr.c_str(), (unsigned int)_eKind);
    throw std::runtime_error(errMsg);
  }

  if (m_mapNameToId.find(_sKindStr) != m_mapNameToId.end()) {
    std::string errMsg = XUtil::format("ERROR: Attempting to register: (%d : %s). Constructor name '%s' already registered to eKind (%d).",
                                       (unsigned int)_eKind, _sKindStr.c_str(),
                                       _sKindStr.c_str(), (unsigned int)m_mapNameToId[_sKindStr]);
    throw std::runtime_error(errMsg);
  }

  if (!_sHeaderJSONName.empty()) {
    if (m_mapJSONNameToKind.find(_sHeaderJSONName) != m_mapJSONNameToKind.end()) {
      std::string errMsg = XUtil::format("ERROR: Attempting to register: (%d : %s). JSON mapping name '%s' already registered to eKind (%d).",
                                         (unsigned int)_eKind, _sKindStr.c_str(),
                                         _sHeaderJSONName.c_str(), (unsigned int)m_mapJSONNameToKind[_sHeaderJSONName]);
      throw std::runtime_error(errMsg);
    }
    m_mapJSONNameToKind[_sHeaderJSONName] = _eKind;
  }

  
  // At this point we know we are good, lets initialize the arrays
  m_mapIdToName[_eKind] = _sKindStr;
  m_mapNameToId[_sKindStr] = _eKind;
  m_mapIdToCtor[_eKind] = _Section_factory;
  m_mapIdToSubSectionSupport[_eKind] = _bSupportsSubSections;
  m_mapIdToSectionIndexSupport[_eKind] = _bSupportsIndexing;
}

bool
Section::translateSectionKindStrToKind(const std::string &_sKindStr, enum axlf_section_kind &_eKind)
{
  if (m_mapNameToId.find(_sKindStr) == m_mapNameToId.end()) {
    return false;   
  }
  _eKind = m_mapNameToId[_sKindStr];
  return true;
}

bool
Section::supportsSubSections(enum axlf_section_kind &_eKind)
{
  if (m_mapIdToSubSectionSupport.find(_eKind) == m_mapIdToSubSectionSupport.end()) {
    return false;   
  }
  return m_mapIdToSubSectionSupport[_eKind];
}

bool
Section::supportsSectionIndex(enum axlf_section_kind &_eKind)
{
  if (m_mapIdToSectionIndexSupport.find(_eKind) == m_mapIdToSectionIndexSupport.end()) {
    return false;   
  }
  return m_mapIdToSectionIndexSupport[_eKind];
}

// -------------------------------------------------------------------------

const std::string & 
Section::getSectionIndexName() const
{
  return m_sIndexName;
}

enum Section::FormatType 
Section::getFormatType(const std::string _sFormatType)
{
  std::string sFormatType = _sFormatType;

  boost::to_upper(sFormatType);

  if (sFormatType == "") { return FT_UNDEFINED; }
  if (sFormatType == "RAW") { return FT_RAW; }
  if (sFormatType == "JSON") { return FT_JSON; }
  if (sFormatType == "HTML") { return FT_HTML; }
  if (sFormatType == "TXT") { return FT_TXT; }
  
  return FT_UNKNOWN;
}

bool 
Section::getKindOfJSON(const std::string &_sJSONStr, enum axlf_section_kind &_eKind) {
  if (_sJSONStr.empty() ||
     (m_mapJSONNameToKind.find(_sJSONStr) == m_mapJSONNameToKind.end()) ) {
    return false;
  }

  _eKind = m_mapJSONNameToKind[_sJSONStr];
  return true;
}


Section*
Section::createSectionObjectOfKind( enum axlf_section_kind _eKind, 
                                    const std::string _sIndexName) {
  Section* pSection = nullptr;

  if (m_mapIdToCtor.find(_eKind) == m_mapIdToCtor.end()) {
    std::string errMsg = XUtil::format("ERROR: Section constructor for the archive section ID '%d' does not exist.  This error is most likely the result of examining a newer version of an archive image then this version of software supports.", (unsigned int)_eKind);
    throw std::runtime_error(errMsg);
  }

  pSection = m_mapIdToCtor[_eKind]();
  pSection->m_eKind = _eKind;
  pSection->m_sKindName = m_mapIdToName[_eKind];
  pSection->m_sIndexName = _sIndexName;

  XUtil::TRACE(XUtil::format("Created segment: %s (%d), index: '%s'",
                             pSection->getSectionKindAsString().c_str(),
                             (unsigned int)pSection->getSectionKind(),
                             pSection->getSectionIndexName().c_str()));
  return pSection;
}


enum axlf_section_kind
Section::getSectionKind() const {
  return m_eKind;
}

const std::string&
Section::getSectionKindAsString() const {
  return m_sKindName;
}

std::string
Section::getName() const {
  return m_name;
}

unsigned int
Section::getSize() const {
  return m_bufferSize;
}

void
Section::initXclBinSectionHeader(axlf_section_header& _sectionHeader) {
  _sectionHeader.m_sectionKind = m_eKind;
  _sectionHeader.m_sectionSize = m_bufferSize;
  XUtil::safeStringCopy((char*)&_sectionHeader.m_sectionName, m_name, sizeof(axlf_section_header::m_sectionName));
}

void
Section::writeXclBinSectionBuffer(std::fstream& _ostream) const
{
  if ((m_pBuffer == nullptr) ||
      (m_bufferSize == 0)) {
    return;
  }

  _ostream.write(m_pBuffer, m_bufferSize);
}

void
Section::readXclBinBinary(std::fstream& _istream, const axlf_section_header& _sectionHeader) {
  // Some error checking
  if ((enum axlf_section_kind)_sectionHeader.m_sectionKind != getSectionKind()) {
    std::string errMsg = XUtil::format("ERROR: Unexpected section kind.  Expected: %d, Read: %d", getSectionKind(), _sectionHeader.m_sectionKind);
    throw std::runtime_error(errMsg);
  }

  if (m_pBuffer != nullptr) {
    std::string errMsg = "ERROR: Binary buffer already exists.";
    throw std::runtime_error(errMsg);
  }

  m_name = (char*)&_sectionHeader.m_sectionName;

  if (_sectionHeader.m_sectionSize > UINT64_MAX) {
    std::string errMsg ("FATAL ERROR: Section header size exceeds internal representation size.");
    throw std::runtime_error(errMsg);
  }

  m_bufferSize = (unsigned int) _sectionHeader.m_sectionSize;

  m_pBuffer = new char[m_bufferSize];

  _istream.seekg(_sectionHeader.m_sectionOffset);

  _istream.read(m_pBuffer, m_bufferSize);

  if (_istream.gcount() != (std::streamsize) m_bufferSize) {
    std::string errMsg = "ERROR: Input stream for the binary buffer is smaller then the expected size.";
    throw std::runtime_error(errMsg);
  }

  XUtil::TRACE(XUtil::format("Section: %s (%d)", getSectionKindAsString().c_str(), (unsigned int)getSectionKind()));
  XUtil::TRACE(XUtil::format("  m_name: %s", m_name.c_str()));
  XUtil::TRACE(XUtil::format("  m_size: %ld", m_bufferSize));
}


void 
Section::readJSONSectionImage(const boost::property_tree::ptree& _ptSection)
{
  std::ostringstream buffer;
  marshalFromJSON(_ptSection, buffer);

  // -- Read contents into memory buffer --
  m_bufferSize = (unsigned int) buffer.tellp();

  if (m_bufferSize == 0) {
    std::string errMsg = XUtil::format("WARNING: Section '%s' content is empty.  No data in the given JSON file.", getSectionKindAsString().c_str());
    std::cout << errMsg.c_str() << std::endl;
    return;
  }

  m_pBuffer = new char[m_bufferSize];
  memcpy(m_pBuffer, buffer.str().c_str(), m_bufferSize);
}

void
Section::readXclBinBinary(std::fstream& _istream,
                          const boost::property_tree::ptree& _ptSection) {
  // Some error checking
  enum axlf_section_kind eKind = (enum axlf_section_kind)_ptSection.get<unsigned int>("Kind");

  if (eKind != getSectionKind()) {
    std::string errMsg = XUtil::format("ERROR: Unexpected section kind.  Expected: %d, Read: %d", getSectionKind(), eKind);
  }

  if (m_pBuffer != nullptr) {
    std::string errMsg = "ERROR: Binary buffer already exists.";
    throw std::runtime_error(errMsg);
  }

  m_name = _ptSection.get<std::string>("Name");


  boost::optional<const boost::property_tree::ptree&> ptPayload = _ptSection.get_child_optional("payload");

  if (ptPayload.is_initialized()) {
    XUtil::TRACE(XUtil::format("Reading in the section '%s' (%d) via metadata.", getSectionKindAsString().c_str(), (unsigned int)getSectionKind()));
    readJSONSectionImage(ptPayload.get());
  } else {
    // We don't initialize the buffer via any metadata.  Just read in the section as is
    XUtil::TRACE(XUtil::format("Reading in the section '%s' (%d) as a image.", getSectionKindAsString().c_str(), (unsigned int)getSectionKind()));

    uint64_t imageSize = XUtil::stringToUInt64(_ptSection.get<std::string>("Size"));
    if (imageSize > UINT64_MAX) {
      std::string errMsg ("FATAL ERROR: Image size exceeds internal representation size.");
      throw std::runtime_error(errMsg);
    }

    m_bufferSize = (unsigned int) imageSize;
    m_pBuffer = new char[m_bufferSize];

    uint64_t offset = XUtil::stringToUInt64(_ptSection.get<std::string>("Offset"));

    _istream.seekg(offset);
    _istream.read(m_pBuffer, m_bufferSize);

    if (_istream.gcount() != (std::streamsize) m_bufferSize) {
      std::string errMsg = "ERROR: Input stream for the binary buffer is smaller then the expected size.";
      throw std::runtime_error(errMsg);
    }
  }

  XUtil::TRACE(XUtil::format("Adding Section: %s (%d)", getSectionKindAsString().c_str(), (unsigned int)getSectionKind()));
  XUtil::TRACE(XUtil::format("  m_name: %s", m_name.c_str()));
  XUtil::TRACE(XUtil::format("  m_size: %ld", m_bufferSize));
}


void
Section::getPayload(boost::property_tree::ptree& _pt) const {
  marshalToJSON(m_pBuffer, m_bufferSize, _pt);
}

void
Section::marshalToJSON(char* _pDataSegment,
                       unsigned int _segmentSize,
                       boost::property_tree::ptree& _ptree) const {
  // Do nothing
}



void 
Section::appendToSectionMetadata(const boost::property_tree::ptree& _ptAppendData,
                                 boost::property_tree::ptree& _ptToAppendTo)
{
   std::string errMsg = "ERROR: The Section '" + getSectionKindAsString() + "' does not support appending metadata";
   throw std::runtime_error(errMsg);
}


void
Section::marshalFromJSON(const boost::property_tree::ptree& _ptSection,
                         std::ostringstream& _buf) const {
  XUtil::TRACE_PrintTree("Payload", _ptSection);
  std::string errMsg = XUtil::format("ERROR: Section '%s' (%d) missing payload parser.", getSectionKindAsString().c_str(), (unsigned int)getSectionKind());
  throw std::runtime_error(errMsg);
}


void 
Section::readPayload(std::fstream& _istream, enum FormatType _eFormatType)
{
    switch (_eFormatType) {
    case FT_RAW:
      {
        axlf_section_header sectionHeader = axlf_section_header {0};
        sectionHeader.m_sectionKind = getSectionKind();
        sectionHeader.m_sectionOffset = 0;
        _istream.seekg(0, _istream.end);
        sectionHeader.m_sectionSize = _istream.tellg();

        readXclBinBinary(_istream, sectionHeader);
        break;
      }
    case FT_JSON:
      {
        // Bring the file into memory
        _istream.seekg(0, _istream.end);
        unsigned int fileSize = (unsigned int) _istream.tellg();

        std::unique_ptr<unsigned char> memBuffer(new unsigned char[fileSize]);
        _istream.clear();
        _istream.seekg(0);
        _istream.read((char*)memBuffer.get(), fileSize);

        XUtil::TRACE_BUF("Buffer", (char*)memBuffer.get(), fileSize);

        // Convert the JSON file to a boost property tree
        std::stringstream ss;
        ss.write((char*) memBuffer.get(), fileSize);

        boost::property_tree::ptree pt;
        boost::property_tree::read_json(ss, pt);

        // O.K. - Lint checking is done and write it to our buffer
        readJSONSectionImage(pt);
        break;
      }
    case FT_HTML:
      // Do nothing
      break;
    case FT_TXT:
      // Do nothing
      break;
    case FT_UNKNOWN:
      // Do nothing
      break;
    case FT_UNDEFINED:
      // Do nothing
      break;
    }
}




void 
Section::readXclBinBinary(std::fstream& _istream, enum FormatType _eFormatType)
{
  switch (_eFormatType) {
  case FT_RAW:
    {
      axlf_section_header sectionHeader = axlf_section_header {0};
      sectionHeader.m_sectionKind = getSectionKind();
      sectionHeader.m_sectionOffset = 0;
      _istream.seekg(0, _istream.end);
      sectionHeader.m_sectionSize = _istream.tellg();

      readXclBinBinary(_istream, sectionHeader);
      break;
    }
  case FT_JSON:
    {
      // Bring the file into memory
      _istream.seekg(0, _istream.end);
      unsigned int fileSize = (unsigned int) _istream.tellg();

      std::unique_ptr<unsigned char> memBuffer(new unsigned char[fileSize]);
      _istream.clear();
      _istream.seekg(0);
      _istream.read((char*)memBuffer.get(), fileSize);

      XUtil::TRACE_BUF("Buffer", (char*)memBuffer.get(), fileSize);

      // Convert the JSON file to a boost property tree
      std::stringstream ss;
      ss.write((char*) memBuffer.get(), fileSize);

      boost::property_tree::ptree pt;
      boost::property_tree::read_json(ss, pt);

      readXclBinBinary(_istream, pt);
      break;
    }
  case FT_HTML:
    // Do nothing
    break;
  case FT_TXT:
    // Do nothing
    break;
  case FT_UNKNOWN:
    // Do nothing
    break;
  case FT_UNDEFINED:
    // Do nothing
    break;
  }
}


void 
Section::dumpContents(std::fstream& _ostream, enum FormatType _eFormatType) const
{
  switch (_eFormatType) {
  case FT_RAW:
    {
      writeXclBinSectionBuffer(_ostream);
      break;
    }
  case FT_JSON:
    {
      boost::property_tree::ptree pt;
      marshalToJSON(m_pBuffer, m_bufferSize, pt);

      boost::property_tree::write_json(_ostream, pt, true /*Pretty print*/);
      break;
    }
  case FT_HTML:
    {
      boost::property_tree::ptree pt;
      marshalToJSON(m_pBuffer, m_bufferSize, pt);

      _ostream << XUtil::format("<!DOCTYPE html><html><body><h1>Section: %s (%d)</h1><pre>", getSectionKindAsString().c_str(), getSectionKind()) << std::endl;
      boost::property_tree::write_json(_ostream, pt, true /*Pretty print*/);
      _ostream << "</pre></body></html>" << std::endl;
      break;
    }
  case FT_UNKNOWN:
    // Do nothing;
    break;
  case FT_TXT:
    // Do nothing;
    break;
  case FT_UNDEFINED:
    break;
  }
}

void 
Section::dumpSubSection(std::fstream& _ostream, 
                        std::string _sSubSection, 
                        enum FormatType _eFormatType) const
{
  writeSubPayload(_sSubSection, _eFormatType, _ostream);
}


void 
Section::printHeader(std::ostream &_ostream) const
{
  _ostream << "Section Header\n";
  _ostream << "  Type    : '" << getSectionKindAsString() << "'" << std::endl;
  _ostream << "  Name    : '" << getName() << "'" << std::endl;
  _ostream << "  Size    : '" << getSize() << "' bytes" << std::endl;
}

bool 
Section::doesSupportAddFormatType(FormatType _eFormatType) const
{
  if (_eFormatType == FT_RAW) {
    return true;
  }
  return false;
}

bool 
Section::doesSupportDumpFormatType(FormatType _eFormatType) const
{
  if (_eFormatType == FT_RAW) {
    return true;
  }
  return false;
}

bool 
Section::supportsSubSection(const std::string &_sSubSectionName) const
{
  return false;
}

bool 
Section::getSubPayload(std::ostringstream &_buf, 
                       const std::string _sSubSection, 
                       enum Section::FormatType _eFormatType) const
{
  // Make sure we support this subsection
  if (supportsSubSection(_sSubSection) == false) {
    return false;
  }

  // Make sure we support the format type
  if (_eFormatType != FT_RAW) {
    return false;
  }

  // All is good now get the data from the section
  getSubPayload(m_pBuffer, m_bufferSize, _buf, _sSubSection, _eFormatType);

  if (_buf.tellp() == 0) {
    return false;
  }

  return true;
}

void 
Section::getSubPayload(char* _pDataSection, 
                       unsigned int _sectionSize, 
                       std::ostringstream &_buf, 
                       const std::string &_sSubSection, 
                       enum Section::FormatType _eFormatType) const
{
  // Empty
}

void
Section::readSubPayload(std::fstream& _istream, 
                        const std::string & _sSubSection, 
                        enum Section::FormatType _eFormatType)
{
  // Make sure we support this subsection
  if (supportsSubSection(_sSubSection) == false) {
    return;
  }

  // All is good now get the data from the section
  std::ostringstream buffer;
  readSubPayload(m_pBuffer, m_bufferSize, _istream, _sSubSection, _eFormatType, buffer);

  // Now for some how cleaning
  if (m_pBuffer != nullptr) {
    delete m_pBuffer;
    m_pBuffer = nullptr;
    m_bufferSize = 0;
  }

  m_bufferSize = (unsigned int) buffer.tellp();

  if (m_bufferSize == 0) {
    std::string errMsg = XUtil::format("WARNING: Section '%s' content is empty.", getSectionKindAsString().c_str());
    throw std::runtime_error(errMsg);
  }

  m_pBuffer = new char[m_bufferSize];
  memcpy(m_pBuffer, buffer.str().c_str(), m_bufferSize);
}

void 
Section::readSubPayload(const char* _pOrigDataSection, 
                        unsigned int _origSectionSize,  
                        std::fstream& _istream, 
                        const std::string & _sSubSection, 
                        enum Section::FormatType _eFormatType, 
                        std::ostringstream &_buffer) const
{
   std::string errMsg = XUtil::format("FATAL ERROR: Section '%s' virtual method readSubPayLoad() not defined.", getSectionKindAsString().c_str());
   throw std::runtime_error(errMsg);
}

bool 
Section::subSectionExists(const std::string &_sSubSectionName) const
{
  return false;
}

void 
Section::writeSubPayload(const std::string & _sSubSectionName, 
                         FormatType _eFormatType, 
                         std::fstream&  _oStream) const
{
  std::string errMsg = XUtil::format("FATAL ERROR: Section '%s' virtual method writeSubPayload() not defined.", getSectionKindAsString().c_str());
  throw std::runtime_error(errMsg);
}


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

// ------ I N C L U D E   F I L E S -------------------------------------------
#include "XclBinClass.h"
#include "Section.h"

#include <stdexcept>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/uuid/uuid.hpp>          // for uuid
#include <boost/uuid/uuid_io.hpp>       // for to_string
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/filesystem.hpp>

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

#include "FormattedOutput.h"
// Generated include files
#include <version.h>
static const std::string MIRROR_DATA_START = "XCLBIN_MIRROR_DATA_START";
static const std::string MIRROR_DATA_END = "XCLBIN_MIRROR_DATA_END";

template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt, 
                         boost::property_tree::ptree::key_type const& key)
{
    std::vector<T> r;
    for (auto& item : pt.get_child(key))
        r.push_back(item.second);
    return r;
}


static
bool getVersionMajorMinorPath(const char * _pVersion, uint8_t & _major, uint8_t & _minor, uint16_t & _patch)
{
  std::string sVersion(_pVersion);
  std::vector<std::string> tokens;
  boost::split(tokens, sVersion, boost::is_any_of("."));
  if ( tokens.size() == 1 ) {
    _major = 0;
    _minor = 0;
    _patch = (uint16_t) std::stoi(tokens[0]);
    return true;
  } 

  if ( tokens.size() == 3 ) {
    _major = (uint8_t) std::stoi(tokens[0]);
    _minor = (uint8_t) std::stoi(tokens[1]);
    _patch = (uint16_t) std::stoi(tokens[2]);
    return true;
  } 

  return false;
}

XclBin::XclBin()
    : m_xclBinHeader({ 0 })
    , m_SchemaVersionMirrorWrite({ 1, 0, 0 }) {
  initializeHeader( m_xclBinHeader);
}

XclBin::~XclBin() {
  for (size_t index = 0; index < m_sections.size(); index++) {
    delete m_sections[index];
  }
  m_sections.clear();
}


void 
XclBin::initializeHeader(axlf &_xclBinHeader)
{
  _xclBinHeader = {0};

  std::string sMagic = "xclbin2";
  XUtil::safeStringCopy(_xclBinHeader.m_magic, sMagic, sizeof(_xclBinHeader.m_magic));
  _xclBinHeader.m_signature_length = -1;  // Initialize to 0xFFs
  memset( _xclBinHeader.reserved, 0xFF, sizeof(_xclBinHeader.reserved) );
  memset( _xclBinHeader.m_keyBlock, 0xFF, sizeof(_xclBinHeader.m_keyBlock) );
  _xclBinHeader.m_uniqueId = time( nullptr );
  _xclBinHeader.m_header.m_timeStamp = time( nullptr );

  // Now populate the version information
  getVersionMajorMinorPath(xrt_build_version, 
                           _xclBinHeader.m_header.m_versionMajor, 
                           _xclBinHeader.m_header.m_versionMinor, 
                           _xclBinHeader.m_header.m_versionPatch);
}

void
XclBin::printSections(std::ostream &_ostream) const {
  XUtil::TRACE("Printing Section Header(s)");
  for (Section *pSection : m_sections) {
    pSection->printHeader(_ostream);
  }
}

void
XclBin::readXclBinBinaryHeader(std::fstream& _istream) {
  // Read in the buffer
  const unsigned int expectBufferSize = sizeof(axlf);

  _istream.seekg(0);
  _istream.read((char*)&m_xclBinHeader, sizeof(axlf));

  if (_istream.gcount() != expectBufferSize) {
    std::string errMsg = "ERROR: Input stream is smaller than the expected header size.";
    throw std::runtime_error(errMsg);
  }

  if (FormattedOutput::getMagicAsString(m_xclBinHeader).c_str() != std::string("xclbin2")) {
    std::string errMsg = "ERROR: The XCLBIN appears to be corrupted (header start key value is not what is expected).";
    throw std::runtime_error(errMsg);
  }
}

void
XclBin::readXclBinBinarySections(std::fstream& _istream) {
  // Read in each section
  unsigned int numberOfSections = m_xclBinHeader.m_header.m_numSections;

  for (unsigned int index = 0; index < numberOfSections; ++index) {
    XUtil::TRACE(XUtil::format("Examining Section: %d of %d", index + 1, m_xclBinHeader.m_header.m_numSections));
    // Find the section header data
    long long sectionOffset = sizeof(axlf) + (index * sizeof(axlf_section_header)) - sizeof(axlf_section_header);
    _istream.seekg(sectionOffset);

    // Read in the section header
    axlf_section_header sectionHeader = axlf_section_header {0};
    const unsigned int expectBufferSize = sizeof(axlf_section_header);

    _istream.read((char*)&sectionHeader, sizeof(axlf_section_header));

    if (_istream.gcount() != expectBufferSize) {
      std::string errMsg = "ERROR: Input stream is smaller than the expected section header size.";
      throw std::runtime_error(errMsg);
    }

    Section* pSection = Section::createSectionObjectOfKind((enum axlf_section_kind)sectionHeader.m_sectionKind);

    // Here for testing purposes, when all segments are supported it should be removed
    if (pSection != nullptr) {
      pSection->readXclBinBinary(_istream, sectionHeader);
      addSection(pSection);
    }
  }
}

void
XclBin::readXclBinBinary(const std::string &_binaryFileName,
                         bool _bMigrate) {
  // Error checks
  if (_binaryFileName.empty()) {
    std::string errMsg = "ERROR: Missing file name to read from.";
    throw std::runtime_error(errMsg);
  }

  // Open the file for consumption
  XUtil::TRACE("Reading xclbin binary file: " + _binaryFileName);
  std::fstream ifXclBin;
  ifXclBin.open(_binaryFileName, std::ifstream::in | std::ifstream::binary);
  if (!ifXclBin.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + _binaryFileName;
    throw std::runtime_error(errMsg);
  }

  if (_bMigrate) {
    boost::property_tree::ptree pt_mirrorData;
    findAndReadMirrorData(ifXclBin, pt_mirrorData);

    // Read in the mirror image
    readXclBinaryMirrorImage(ifXclBin, pt_mirrorData);
  } else {
    // Read in the header
    readXclBinBinaryHeader(ifXclBin);

    // Read the sections
    readXclBinBinarySections(ifXclBin);
  }

  ifXclBin.close();
}

void
XclBin::addHeaderMirrorData(boost::property_tree::ptree& _pt_header) {
  XUtil::TRACE("Creating Header Mirror ptree");

  // Axlf structure
  {
    _pt_header.put("Magic", FormattedOutput::getMagicAsString(m_xclBinHeader).c_str());
    _pt_header.put("SignatureLength", FormattedOutput::getSignatureLengthAsString(m_xclBinHeader).c_str());
    _pt_header.put("KeyBlock", FormattedOutput::getKeyBlockAsString(m_xclBinHeader).c_str());
    _pt_header.put("UniqueID", FormattedOutput::getUniqueIdAsString(m_xclBinHeader).c_str());
  }

  // Axlf_header structure
  {
    _pt_header.put("TimeStamp", FormattedOutput::getTimeStampAsString(m_xclBinHeader).c_str());
    _pt_header.put("FeatureRomTimeStamp", FormattedOutput::getFeatureRomTimeStampAsString(m_xclBinHeader).c_str());
    _pt_header.put("Version", FormattedOutput::getVersionAsString(m_xclBinHeader).c_str());
    _pt_header.put("Mode", FormattedOutput::getModeAsString(m_xclBinHeader).c_str());
    _pt_header.put("FeatureRomUUID", FormattedOutput::getFeatureRomUuidAsString(m_xclBinHeader).c_str());
    _pt_header.put("PlatformVBNV", FormattedOutput::getPlatformVbnvAsString(m_xclBinHeader).c_str());
    _pt_header.put("XclBinUUID", FormattedOutput::getXclBinUuidAsString(m_xclBinHeader).c_str());
    _pt_header.put("DebugBin", FormattedOutput::getDebugBinAsString(m_xclBinHeader).c_str());
  }
}


void
XclBin::writeXclBinBinaryHeader(std::fstream& _ostream, boost::property_tree::ptree& _mirroredData) {
  // Write the header (minus the section header array)
  XUtil::TRACE("Writing xclbin binary header");
  _ostream.write((char*)&m_xclBinHeader, sizeof(axlf) - sizeof(axlf_section_header));

  // Get mirror data
  boost::property_tree::ptree pt_header;
  addHeaderMirrorData(pt_header);

  _mirroredData.add_child("header", pt_header);
}


void
XclBin::writeXclBinBinarySections(std::fstream& _ostream, boost::property_tree::ptree& _mirroredData) {
  // Nothing to write
  if (m_sections.empty()) {
    return;
  }

  // Prepare the array
  // WARNING - Do not us this code in Linux.  It hasn't been fully flushed out.
  struct axlf_section_header *sectionHeader = new struct axlf_section_header[m_sections.size()];
  memset(sectionHeader, 0, sizeof(struct axlf_section_header) * m_sections.size());  // Zero out memory

  // Populate the array size and offsets
  uint64_t currentOffset = (uint64_t) (sizeof(axlf) - sizeof(axlf_section_header) + (sizeof(axlf_section_header) * m_sections.size()));

  for (unsigned int index = 0; index < m_sections.size(); ++index) {
    // Calculate padding
    currentOffset += (uint64_t) XUtil::bytesToAlign(currentOffset);

    // Initialize section header
    m_sections[index]->initXclBinSectionHeader(sectionHeader[index]);
    sectionHeader[index].m_sectionOffset = currentOffset;
    currentOffset += (uint64_t) sectionHeader[index].m_sectionSize;
  }

  XUtil::TRACE("Writing xclbin section header array");
  _ostream.write((char*) sectionHeader, sizeof(axlf_section_header) * m_sections.size());

  // Write out each of the sections
  for (unsigned int index = 0; index < m_sections.size(); ++index) {
    XUtil::TRACE(XUtil::format("Writing section: Index: %d, ID: %d", index, sectionHeader[index].m_sectionKind));

    // Align section to next 8 byte boundary
    unsigned int runningOffset = (unsigned int) _ostream.tellp();
    unsigned int bytePadding = XUtil::bytesToAlign(runningOffset);
    if (bytePadding != 0) {
      static char holePack[] = { (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0 };
      _ostream.write(holePack, bytePadding);
    }
    runningOffset += bytePadding;

    // Check current and expected offsets
    if (runningOffset != sectionHeader[index].m_sectionOffset) {
      std::string errMsg = XUtil::format("ERROR: Expected offset (0x%lx) does not match actual (0x%lx)", sectionHeader[index].m_sectionOffset, runningOffset);
      throw std::runtime_error(errMsg);
    }

    // Write buffer
    m_sections[index]->writeXclBinSectionBuffer(_ostream);

    // Write mirror data
    {
      XUtil::TRACE("");
      XUtil::TRACE(XUtil::format("Adding mirror properties[%d]", index));

      boost::property_tree::ptree pt_sectionHeader;

      XUtil::TRACE(XUtil::format("Kind: %d, Name: %s, Offset: 0x%lx, Size: 0x%lx",
                                 sectionHeader[index].m_sectionKind,
                                 sectionHeader[index].m_sectionName,
                                 sectionHeader[index].m_sectionOffset,
                                 sectionHeader[index].m_sectionSize));

      pt_sectionHeader.put("Kind", XUtil::format("%d", sectionHeader[index].m_sectionKind).c_str());
      pt_sectionHeader.put("Name", XUtil::format("%s", sectionHeader[index].m_sectionName).c_str());
      pt_sectionHeader.put("Offset", XUtil::format("0x%lx", sectionHeader[index].m_sectionOffset).c_str());
      pt_sectionHeader.put("Size", XUtil::format("0x%lx", sectionHeader[index].m_sectionSize).c_str());

      boost::property_tree::ptree pt_Payload;
      if (m_sections[index]->doesSupportAddFormatType(Section::FT_JSON) && 
          m_sections[index]->doesSupportDumpFormatType(Section::FT_JSON)) {
        m_sections[index]->getPayload(pt_Payload);
      }

      if (pt_Payload.size() != 0) {
        pt_sectionHeader.add_child("payload", pt_Payload);
      }

      _mirroredData.add_child("section_header", pt_sectionHeader);
    }
  }

  delete[] sectionHeader;
}


void
XclBin::writeXclBinBinaryMirrorData(std::fstream& _ostream,
                                    const boost::property_tree::ptree& _mirroredData) const {
  _ostream << MIRROR_DATA_START;
  boost::property_tree::write_json(_ostream, _mirroredData, false /*Pretty print*/);
  _ostream << MIRROR_DATA_END;

  XUtil::TRACE_PrintTree("Mirrored Data", _mirroredData);
}

void
XclBin::updateUUID() {
    static_assert (sizeof(boost::uuids::uuid) == 16, "ERROR: UUID size mismatch");
    static_assert (sizeof(axlf_header::uuid) == 16, "ERROR: UUID size mismatch");

    boost::uuids::uuid uuid = boost::uuids::random_generator()();

    // Copy the values to the UUID structure
    memcpy((void *) &m_xclBinHeader.m_header.uuid, (void *)&uuid, sizeof(axlf_header::rom_uuid));
    XUtil::TRACE("Updated xclbin UUID");
}

void
XclBin::writeXclBinBinary(const std::string &_binaryFileName, 
                          bool _bSkipUUIDInsertion) {
  // Error checks
  if (_binaryFileName.empty()) {
    std::string errMsg = "ERROR: Missing file name to write to.";
    throw std::runtime_error(errMsg);
  }

  // Write the xclbin file image
  XUtil::TRACE("Writing the xclbin binary file: " + _binaryFileName);
  std::fstream ofXclBin;
  ofXclBin.open(_binaryFileName, std::ifstream::out | std::ifstream::binary);
  if (!ofXclBin.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for writing: " + _binaryFileName;
    throw std::runtime_error(errMsg);
  }

  if (_bSkipUUIDInsertion) {
    XUtil::TRACE("Skipping xclbin's UUID insertion.");
  } else {
    updateUUID();
  }

  // Mirrored data
  boost::property_tree::ptree mirroredData;

  // Add Version information
  addPTreeSchemaVersion(mirroredData, m_SchemaVersionMirrorWrite);

  // Write in the header data
  writeXclBinBinaryHeader(ofXclBin, mirroredData);

  // Write the section array and sections
  writeXclBinBinarySections(ofXclBin, mirroredData);

  // Write out our mirror data
  writeXclBinBinaryMirrorData(ofXclBin, mirroredData);

  // Update header file length
  {
    // Determine file size
    ofXclBin.seekg(0, ofXclBin.end);
    unsigned int streamSize = (unsigned int) ofXclBin.tellg();

    // Update Header
    m_xclBinHeader.m_header.m_length = streamSize;

    // Write out the header...again
    ofXclBin.seekg(0, ofXclBin.beg);
    boost::property_tree::ptree dummyData;
    writeXclBinBinaryHeader(ofXclBin, dummyData);
  }

  // Close file
  ofXclBin.close();

  std::cout << XUtil::format("Successfully wrote (%ld bytes) to the output file: %s", m_xclBinHeader.m_header.m_length, _binaryFileName.c_str()).c_str() << std::endl;
}


void
XclBin::addPTreeSchemaVersion(boost::property_tree::ptree& _pt, SchemaVersion const& _schemaVersion) {

  XUtil::TRACE("");
  XUtil::TRACE("Adding Versioning Properties");

  boost::property_tree::ptree pt_schemaVersion;

  XUtil::TRACE(XUtil::format("major: %d, minor: %d, patch: %d",
                             _schemaVersion.major,
                             _schemaVersion.minor,
                             _schemaVersion.patch));

  pt_schemaVersion.put("major", XUtil::format("%d", _schemaVersion.major).c_str());
  pt_schemaVersion.put("minor", XUtil::format("%d", _schemaVersion.minor).c_str());
  pt_schemaVersion.put("patch", XUtil::format("%d", _schemaVersion.patch).c_str());
  _pt.add_child("schema_version", pt_schemaVersion);
}


void
XclBin::getSchemaVersion(boost::property_tree::ptree& _pt, SchemaVersion& _schemaVersion) {
  XUtil::TRACE("SchemaVersion");

  _schemaVersion.major = _pt.get<unsigned int>("major");
  _schemaVersion.minor = _pt.get<unsigned int>("minor");
  _schemaVersion.patch = _pt.get<unsigned int>("patch");

  XUtil::TRACE(XUtil::format("major: %d, minor: %d, patch: %d",
                             _schemaVersion.major,
                             _schemaVersion.minor,
                             _schemaVersion.patch));
}

void
XclBin::findAndReadMirrorData(std::fstream& _istream, boost::property_tree::ptree& _mirrorData) const {
  XUtil::TRACE("Searching for mirrored data...");

  // Find start of buffer
  _istream.seekg(0);
  unsigned int startOffset = 0;
  if (XUtil::findBytesInStream(_istream, MIRROR_DATA_START, startOffset) == true) {
    XUtil::TRACE(XUtil::format("Found MIRROR_DATA_START at offset: 0x%lx", startOffset));
    startOffset += (unsigned int) MIRROR_DATA_START.length();
  }  else {
    std::string errMsg;
    errMsg  = "ERROR: Mirror backup data not found in given file.\n"; 
    errMsg += "       The given archive image does not contain any metadata to\n";
    errMsg += "       migrate the data image to the current format.\n";
    errMsg += "       The lack of metadata is usually the result of attempting\n";
    errMsg += "       to migrate a pre-2018.3 archive.";
                          
    throw std::runtime_error(errMsg);
  }

  // Find end of buffer (continue where we left off)
  _istream.seekg(startOffset);
  unsigned int bufferSize = 0;
  if (XUtil::findBytesInStream(_istream, MIRROR_DATA_END, bufferSize) == true) {
    XUtil::TRACE(XUtil::format("Found MIRROR_DATA_END.  Buffersize: 0x%lx", bufferSize));
  }  else {
    std::string errMsg = "ERROR: Mirror backup data not well formed in given file.";
    throw std::runtime_error(errMsg);
  }

  // Bring the mirror metadata into memory
  std::unique_ptr<unsigned char> memBuffer(new unsigned char[bufferSize]);
  _istream.clear();
  _istream.seekg(startOffset);
  _istream.read((char*)memBuffer.get(), bufferSize);

  XUtil::TRACE_BUF("Buffer", (char*)memBuffer.get(), bufferSize);

  // Convert the JSON file to a boost property tree
  std::stringstream ss;
  ss.write((char*) memBuffer.get(), bufferSize);

  // TODO: Catch the exception (if any) from this call and produce a nice message
  try {
    boost::property_tree::read_json(ss, _mirrorData);
  } catch (boost::property_tree::json_parser_error &e) {
    std::string errMsg = XUtil::format("ERROR: Parsing mirror metadata in the xclbin archive on line %d: %s", e.line(), e.message().c_str());
    throw std::runtime_error(errMsg);
  }

  XUtil::TRACE_PrintTree("Mirror", _mirrorData);
}


void
XclBin::readXclBinHeader(const boost::property_tree::ptree& _ptHeader,
                         struct axlf& _axlfHeader) {
  XUtil::TRACE("Reading via JSON mirror xclbin header information.");
  XUtil::TRACE_PrintTree("Header Mirror Image", _ptHeader);

  // Clear the previous header information
  _axlfHeader = { 0 };

  std::string sMagic = _ptHeader.get<std::string>("Magic");
  XUtil::safeStringCopy((char*)&_axlfHeader.m_magic, sMagic, sizeof(axlf::m_magic));
  _axlfHeader.m_signature_length = _ptHeader.get<int32_t>("SignatureLength", -1);
  std::string sKeyBlock = _ptHeader.get<std::string>("KeyBlock");
  XUtil::hexStringToBinaryBuffer(sKeyBlock, (unsigned char*)&_axlfHeader.m_keyBlock, sizeof(axlf::m_keyBlock));
  _axlfHeader.m_uniqueId = XUtil::stringToUInt64(_ptHeader.get<std::string>("UniqueID"));

  _axlfHeader.m_header.m_timeStamp = XUtil::stringToUInt64(_ptHeader.get<std::string>("TimeStamp"));
  _axlfHeader.m_header.m_featureRomTimeStamp = XUtil::stringToUInt64(_ptHeader.get<std::string>("FeatureRomTimeStamp"));
  std::string sVersion = _ptHeader.get<std::string>("Version");
  getVersionMajorMinorPath(sVersion.c_str(), 
                           _axlfHeader.m_header.m_versionMajor, 
                           _axlfHeader.m_header.m_versionMinor, 
                           _axlfHeader.m_header.m_versionPatch);

  _axlfHeader.m_header.m_mode = _ptHeader.get<uint32_t>("Mode");

  std::string sFeatureRomUUID = _ptHeader.get<std::string>("FeatureRomUUID");
  XUtil::hexStringToBinaryBuffer(sFeatureRomUUID, (unsigned char*)&_axlfHeader.m_header.rom_uuid, sizeof(axlf_header::rom_uuid));
  std::string sPlatformVBNV = _ptHeader.get<std::string>("PlatformVBNV");
  XUtil::safeStringCopy((char*)&_axlfHeader.m_header.m_platformVBNV,
  
                        sPlatformVBNV, sizeof(axlf_header::m_platformVBNV));
  std::string sXclBinUUID = _ptHeader.get<std::string>("XclBinUUID");
  XUtil::hexStringToBinaryBuffer(sXclBinUUID, (unsigned char*)&_axlfHeader.m_header.uuid, sizeof(axlf_header::uuid));

  std::string sDebugBin = _ptHeader.get<std::string>("DebugBin");
  XUtil::safeStringCopy((char*)&_axlfHeader.m_header.m_debug_bin, sDebugBin, sizeof(axlf_header::m_debug_bin));

  XUtil::TRACE("Done Reading via JSON mirror xclbin header information.");
}

void
XclBin::readXclBinSection(std::fstream& _istream,
                          const boost::property_tree::ptree& _ptSection) {
  enum axlf_section_kind eKind = (enum axlf_section_kind)_ptSection.get<unsigned int>("Kind");

  Section* pSection = Section::createSectionObjectOfKind(eKind);

  pSection->readXclBinBinary(_istream, _ptSection);
  addSection(pSection);
}



void
XclBin::readXclBinaryMirrorImage(std::fstream& _istream,
                                 const boost::property_tree::ptree& _mirrorData) {
  // Iterate over each entry
  for (boost::property_tree::ptree::const_iterator ptEntry = _mirrorData.begin();
       ptEntry != _mirrorData.end();
       ++ptEntry) {
    XUtil::TRACE("Processing: '" + ptEntry->first + "'");

    // ---------------------------------------------------------------------
    if (ptEntry->first == "schema_version") {
      XUtil::TRACE("Examining the xclbin version schema");
      // TODO: getSchemaVersion(ptSegment->second, schemaVersion);
      continue;
    }

    // ---------------------------------------------------------------------
    if (ptEntry->first == "header") {
      readXclBinHeader(ptEntry->second, m_xclBinHeader);
      continue;
    }

    // ---------------------------------------------------------------------
    if (ptEntry->first == "section_header") {
      readXclBinSection(_istream, ptEntry->second);
      continue;
    }
    XUtil::TRACE("Skipping unknown section: " + ptEntry->first);
  }
}

void
XclBin::addSection(Section* _pSection) {
  // Error check
  if (_pSection == nullptr) {
    return;
  }

  m_sections.push_back(_pSection);
  m_xclBinHeader.m_header.m_numSections = (uint32_t) m_sections.size();
}

void 
XclBin::removeSection(const Section* _pSection)
{
  // Error check
  if (_pSection == nullptr) {
    return;
  }

  for (unsigned int index = 0; index < m_sections.size(); ++index) {
    if ((void *) m_sections[index] == (void *) _pSection) {
      XUtil::TRACE(XUtil::format("Removing and deleting section '%s' (%d).", _pSection->getSectionKindAsString().c_str(), _pSection->getSectionKind()));
      m_sections.erase(m_sections.begin() + index);
      delete _pSection;
      m_xclBinHeader.m_header.m_numSections = (uint32_t) m_sections.size();
      return;
    }
  }

  std::string errMsg=XUtil::format("ERROR: Section '%s' (%d) not found", _pSection->getSectionKindAsString().c_str(), _pSection->getSectionKind());
  throw XUtil::XclBinUtilException(XET_MISSING_SECTION, errMsg);
}

Section *
XclBin::findSection(enum axlf_section_kind _eKind, 
                    const std::string _indexName)
{
  for (unsigned int index = 0; index < m_sections.size(); ++index) {
    if (m_sections[index]->getSectionKind() == _eKind) {
      if (m_sections[index]->getSectionIndexName().compare(_indexName) == 0) {
        return m_sections[index];
      }
    }
  }
  return nullptr;
}

std::vector<Section*> XclBin::getSections() {
  return m_sections;
}

void 
XclBin::removeSection(const std::string & _sSectionToRemove)
{
  XUtil::TRACE("Removing Section: " + _sSectionToRemove);

  std::string sectionName = _sSectionToRemove;
  std::string sectionIndexName;

  // Extract the section index (if it is there)
  const std::string sectionIndexStartDelimiter = "[";    
  const char sectionIndexEndDelimiter = ']';    
  std::size_t sectionIndex =  _sSectionToRemove.find_first_of(sectionIndexStartDelimiter, 0);

  // Was the start index found?
  if (sectionIndex != std::string::npos) {
    // We need to have an end delimiter
    if (sectionIndexEndDelimiter != _sSectionToRemove.back()) {
      std::string errMsg = XUtil::format("Error: Expected format <section>[<section_index>] when using a section index.  Received: %s.", _sSectionToRemove.c_str());
      throw std::runtime_error(errMsg);
    }

    // Extract the index name
    sectionIndexName = _sSectionToRemove.substr(sectionIndex + 1);
    sectionIndexName.pop_back();  // Remove ']'

    // Extract the section name
    sectionName = _sSectionToRemove.substr(0, sectionIndex);
  }

  enum axlf_section_kind _eKind;
  
  if (Section::translateSectionKindStrToKind(sectionName, _eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name.", sectionName.c_str());
    throw std::runtime_error(errMsg);
  }

  if ((Section::supportsSectionIndex(_eKind) == true) && 
      (sectionIndexName.empty())) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' can only be deleted with indexes.", sectionName.c_str());
    throw std::runtime_error(errMsg);
  }

  if ((Section::supportsSectionIndex(_eKind) == false) && 
      (!sectionIndexName.empty())) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' cannot be deleted with index values (not supported).", sectionName.c_str());
    throw std::runtime_error(errMsg);
  }

  const Section * pSection = findSection(_eKind, sectionIndexName);
  if (pSection == nullptr) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' is not part of the xclbin archive.", _sSectionToRemove.c_str());
    throw XUtil::XclBinUtilException(XET_MISSING_SECTION, errMsg);
  }

  removeSection(pSection);

  std::string indexEntry;
  if (!sectionIndexName.empty()) {
    indexEntry = "[" + sectionIndexName + "]";
  }
  std::cout << std::endl << XUtil::format("Section '%s%s'(%d) was successfully removed",
                                          pSection->getSectionKindAsString().c_str(), 
                                          indexEntry.c_str(),
                                          pSection->getSectionKind()) << std::endl;
}


void 
XclBin::replaceSection(ParameterSectionData &_PSD)
{
  enum axlf_section_kind eKind;
  if (Section::translateSectionKindStrToKind(_PSD.getSectionName(), eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name.", _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  Section *pSection = findSection(eKind);
  if (pSection == nullptr) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' does not exist.", _PSD.getSectionName().c_str());
    throw XUtil::XclBinUtilException(XET_MISSING_SECTION, errMsg);
  }

  std::string sSectionFileName = _PSD.getFile();
  // Write the xclbin file image
  std::fstream iSectionFile;
  iSectionFile.open(sSectionFileName, std::ifstream::in | std::ifstream::binary);
  if (!iSectionFile.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + sSectionFileName;
    throw std::runtime_error(errMsg);
  }

  pSection->purgeBuffers();

  pSection->readPayload(iSectionFile, _PSD.getFormatType());

  updateHeaderFromSection(pSection);

  boost::filesystem::path p(sSectionFileName);
  std::string sBaseName = p.stem().string();
  pSection->setName(sBaseName);

  XUtil::TRACE(XUtil::format("Section '%s' (%d) successfully added.", pSection->getSectionKindAsString().c_str(), pSection->getSectionKind()));
  std::cout << std::endl << XUtil::format("Section: '%s'(%d) was successfully added.\nSize   : %ld bytes\nFormat : %s\nFile   : '%s'", 
                                          pSection->getSectionKindAsString().c_str(), pSection->getSectionKind(),
                                          pSection->getSize(),
                                          _PSD.getFormatTypeAsStr().c_str(), sSectionFileName.c_str()).c_str() << std::endl;
}

void
XclBin::updateHeaderFromSection(Section *_pSection)
{
  if (_pSection == nullptr) {
    return;
  }

  if (_pSection->getSectionKind() == BUILD_METADATA) {
    boost::property_tree::ptree pt;
    _pSection->getPayload(pt);

    boost::property_tree::ptree ptDsa;
    ptDsa = pt.get_child("build_metadata.dsa");

    std::vector<boost::property_tree::ptree> feature_roms = as_vector<boost::property_tree::ptree>(ptDsa, "feature_roms");

    boost::property_tree::ptree featureRom;
    if (!feature_roms.empty()) {
      featureRom = feature_roms[0];
    }

    // Feature ROM Time Stamp
    m_xclBinHeader.m_header.m_featureRomTimeStamp = XUtil::stringToUInt64(featureRom.get<std::string>("timeSinceEpoch", "0"));

    // Feature ROM UUID
    std::string sFeatureRomUUID = featureRom.get<std::string>("uuid", "00000000000000000000000000000000");
    sFeatureRomUUID.erase(std::remove(sFeatureRomUUID.begin(), sFeatureRomUUID.end(), '-'), sFeatureRomUUID.end()); // Remove the '-'
    XUtil::hexStringToBinaryBuffer(sFeatureRomUUID, (unsigned char*)&m_xclBinHeader.m_header.rom_uuid, sizeof(axlf_header::rom_uuid));

    // Feature ROM VBNV
    std::string sPlatformVBNV = featureRom.get<std::string>("vbnvName", "");
    XUtil::safeStringCopy((char*)&m_xclBinHeader.m_header.m_platformVBNV, sPlatformVBNV, sizeof(axlf_header::m_platformVBNV));

    // Examine OLD names -- // This code can be removed AFTER xocc has been updated to use the new format
    {
      // Feature ROM Time Stamp
      if (m_xclBinHeader.m_header.m_featureRomTimeStamp == 0) {
        m_xclBinHeader.m_header.m_featureRomTimeStamp = XUtil::stringToUInt64(featureRom.get<std::string>("time_epoch", "0"));
      }
    
      // Feature ROM VBNV
      if (sPlatformVBNV.empty()) {
        sPlatformVBNV = featureRom.get<std::string>("vbnv_name", "");
        XUtil::safeStringCopy((char*)&m_xclBinHeader.m_header.m_platformVBNV, sPlatformVBNV, sizeof(axlf_header::m_platformVBNV));
      }
    }

    XUtil::TRACE_PrintTree("Build MetaData To Be examined", pt);
  }
}

void 
XclBin::addSubSection(ParameterSectionData &_PSD)
{
  XUtil::TRACE("Add Sub-Section");

  // See if there is a subsection to add
  std::string sSubSection = _PSD.getSubSectionName();
  if (sSubSection.empty()) {
    std::string errMsg = XUtil::format("ERROR: No subsection specified: '%s'", _PSD.getOriginalFormattedString().c_str());
    throw std::runtime_error(errMsg);
  }

  // Get the section kind
  enum axlf_section_kind eKind;
  if (Section::translateSectionKindStrToKind(_PSD.getSectionName(), eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name for the command: %s", _PSD.getSectionName().c_str(), _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  // See if the section support sub-sections
  if (Section::supportsSubSections(eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name.", _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  // Determine if the section already exists
  Section *pSection = findSection(eKind, _PSD.getSectionIndexName());
  bool bNewSection = false;
  if (pSection != nullptr) {
    // Check to see if the subsection is supported
    if (pSection->supportsSubSection(sSubSection) == false) {
      std::string errMsg = XUtil::format("ERROR: Section '%s' does not support the subsection: '%s'", pSection->getSectionKindAsString().c_str(), sSubSection.c_str());
      throw std::runtime_error(errMsg);
    }

    // Check to see if this subsection exists, if so bail
    std::ostringstream buffer;
    if (pSection->subSectionExists(_PSD.getSubSectionName()) == true) {
      std::string errMsg = XUtil::format("ERROR: Section '%s' subsection '%s' already exists", pSection->getSectionKindAsString().c_str(), sSubSection.c_str());
      throw std::runtime_error(errMsg);
    }
  } else {
    pSection = Section::createSectionObjectOfKind(eKind, _PSD.getSectionIndexName());
    bNewSection = true;

    // Check to see if the subsection is supported
    if (pSection->supportsSubSection(sSubSection) == false) {
      std::string errMsg = XUtil::format("ERROR: Section '%s' does not support the subsection: '%s'", pSection->getSectionKindAsString().c_str(), sSubSection.c_str());
      throw std::runtime_error(errMsg);
    }

    boost::filesystem::path p(_PSD.getFile());
    std::string sBaseName = p.stem().string();
    pSection->setName(sBaseName);
  }

  // At this point we know we can add the subsection

  // Open the file to be read.
  std::string sSectionFileName = _PSD.getFile();
  std::fstream iSectionFile;
  iSectionFile.open(sSectionFileName, std::ifstream::in | std::ifstream::binary);
  if (!iSectionFile.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + sSectionFileName;
    throw std::runtime_error(errMsg);
  }

  // Read in the data
  pSection->readSubPayload(iSectionFile, _PSD.getSubSectionName(), _PSD.getFormatType());

  // Clean-up
  if (bNewSection == true) {
    addSection(pSection);
  }
  
  std::string sSectionAddedName = pSection->getSectionKindAsString();

  XUtil::TRACE(XUtil::format("Section '%s-%s' (%d) successfully added.", sSectionAddedName.c_str(), sSubSection.c_str(), pSection->getSectionKind()));
  std::string optionalIndex;
  if (!(pSection->getSectionIndexName().empty())) {
    optionalIndex = XUtil::format("[%s]", pSection->getSectionIndexName().c_str());
  }

  std::cout << std::endl << XUtil::format("Section: '%s%s-%s'(%d) was successfully added.\nSize   : %ld bytes\nFormat : %s\nFile   : '%s'", 
                                          sSectionAddedName.c_str(), 
                                          optionalIndex.c_str(),
                                          sSubSection.c_str(), pSection->getSectionKind(),
                                          pSection->getSize(),
                                          _PSD.getFormatTypeAsStr().c_str(), sSectionFileName.c_str()).c_str() << std::endl;
}


void 
XclBin::addSection(ParameterSectionData &_PSD)
{
  XUtil::TRACE("Add Section");

  // See if the user is attempting to add a sub-section
  if (!_PSD.getSubSectionName().empty()) {
    addSubSection(_PSD);
    return;
  }

  // Get the section kind
  enum axlf_section_kind eKind;
  if (Section::translateSectionKindStrToKind(_PSD.getSectionName(), eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name.", _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  // Open the file to be read.
  std::string sSectionFileName = _PSD.getFile();
  std::fstream iSectionFile;
  iSectionFile.open(sSectionFileName, std::ifstream::in | std::ifstream::binary);
  if (!iSectionFile.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + sSectionFileName;
    throw std::runtime_error(errMsg);
  }

  // Determine if the section already exists
  Section *pSection = findSection(eKind);
  if (pSection != nullptr) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' already exists.", _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  pSection = Section::createSectionObjectOfKind(eKind);

  // Check to see if the given format type is supported
  if (pSection->doesSupportAddFormatType(_PSD.getFormatType()) == false) {
    std::string errMsg = XUtil::format("ERROR: The %s section does not support reading the %s file type.",
                                        pSection->getSectionKindAsString().c_str(),
                                        _PSD.getFormatTypeAsStr().c_str());
    throw std::runtime_error(errMsg);
  }

  // Read in the data
  pSection->readPayload(iSectionFile, _PSD.getFormatType());

  // Post-cleanup
  boost::filesystem::path p(sSectionFileName);
  std::string sBaseName = p.stem().string();
  pSection->setName(sBaseName);

  bool bAllowZeroSize = ((pSection->getSectionKind() == DEBUG_DATA)
      && (_PSD.getFormatType() == Section::FT_RAW));

  if ((!bAllowZeroSize) && (pSection->getSize() == 0)) {
    std::cout << std::endl << XUtil::format("Section: '%s'(%d) was empty.  No action taken.\nFormat : %s\nFile   : '%s'", 
                                          pSection->getSectionKindAsString().c_str(), 
                                          pSection->getSectionKind(),
                                          _PSD.getFormatTypeAsStr().c_str(), sSectionFileName.c_str()).c_str() << std::endl;
    delete pSection;
    pSection = nullptr;
    return;
  }

  addSection(pSection);
  updateHeaderFromSection(pSection);

  std::string sSectionAddedName = pSection->getSectionKindAsString();

  XUtil::TRACE(XUtil::format("Section '%s' (%d) successfully added.", sSectionAddedName.c_str(), pSection->getSectionKind()));
  std::cout << std::endl << XUtil::format("Section: '%s'(%d) was successfully added.\nSize   : %ld bytes\nFormat : %s\nFile   : '%s'", 
                                          sSectionAddedName.c_str(), pSection->getSectionKind(),
                                          pSection->getSize(),
                                          _PSD.getFormatTypeAsStr().c_str(), sSectionFileName.c_str()).c_str() << std::endl;
}


void 
XclBin::addSections(ParameterSectionData &_PSD)
{
  if (!_PSD.getSectionName().empty()) {
    std::string errMsg = "ERROR: Section given for a wildcard JSON section add is not empty.";
    throw std::runtime_error(errMsg);
  }

  if (_PSD.getFormatType() != Section::FT_JSON) {
    std::string errMsg = XUtil::format("ERROR: Expecting JSON format type, got '%s'.", _PSD.getFormatTypeAsStr().c_str());
    throw std::runtime_error(errMsg);
  }

  std::string sJSONFileName = _PSD.getFile();

  std::fstream fs;
  fs.open(sJSONFileName, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + sJSONFileName;
    throw std::runtime_error(errMsg);
  }

  //  Add a new element to the collection and parse the jason file
  XUtil::TRACE("Reading JSON File: '" + sJSONFileName + '"');
  boost::property_tree::ptree pt;
  try {
    boost::property_tree::read_json(fs, pt);
  } catch (boost::property_tree::json_parser_error &e) {
    std::string errMsg = XUtil::format("ERROR: Parsing the file '%s' on line %d: %s", sJSONFileName.c_str(), e.line(), e.message().c_str());
    throw std::runtime_error(errMsg);
  }
  
  XUtil::TRACE("Examining the property tree from the JSON's file: '" + sJSONFileName + "'");
  XUtil::TRACE("Property Tree: Root");
  XUtil::TRACE_PrintTree("Root", pt);

  for (boost::property_tree::ptree::iterator ptSection = pt.begin(); ptSection != pt.end(); ++ptSection) {
    const std::string & sectionName = ptSection->first;
    if (sectionName == "schema_version") {
      XUtil::TRACE("Skipping: '" + sectionName + "'");
      continue;
    }

    XUtil::TRACE("Processing: '" + sectionName + "'");

    enum axlf_section_kind eKind;
    if (Section::getKindOfJSON(sectionName, eKind) == false) {
      std::string errMsg = XUtil::format("ERROR: Unknown JSON section '%s' in file: %s", sectionName.c_str(), sJSONFileName.c_str());
      throw std::runtime_error(errMsg);
    }

    Section *pSection = findSection(eKind);
    if (pSection != nullptr) {
      std::string errMsg = XUtil::format("ERROR: Section '%s' already exists.", pSection->getSectionKindAsString().c_str());
      throw std::runtime_error(errMsg);
    }

    pSection = Section::createSectionObjectOfKind(eKind);
    pSection->readJSONSectionImage(pt);
    if (pSection->getSize() == 0) {
      std::cout << std::endl << XUtil::format("Section: '%s'(%d) was empty.  No action taken.\nFormat : %s\nFile   : '%s'", 
                                            pSection->getSectionKindAsString().c_str(), 
                                            pSection->getSectionKind(),
                                            _PSD.getFormatTypeAsStr().c_str(), sectionName.c_str()).c_str() << std::endl;
      delete pSection;
      pSection = nullptr;
      continue;
    }
    addSection(pSection);
    updateHeaderFromSection(pSection);
    XUtil::TRACE(XUtil::format("Section '%s' (%d) successfully added.", pSection->getSectionKindAsString().c_str(), pSection->getSectionKind()));
    std::cout << std::endl << XUtil::format("Section: '%s'(%d) was successfully added.\nFormat : %s\nFile   : '%s'", 
                                          pSection->getSectionKindAsString().c_str(), 
                                          pSection->getSectionKind(),
                                          _PSD.getFormatTypeAsStr().c_str(), sectionName.c_str()).c_str() << std::endl;
  }
}

void 
XclBin::appendSections(ParameterSectionData &_PSD)
{
  if (!_PSD.getSectionName().empty()) {
    std::string errMsg = "ERROR: Section given for a wildcard JSON section add is not empty.";
    throw std::runtime_error(errMsg);
  }

  if (_PSD.getFormatType() != Section::FT_JSON) {
    std::string errMsg = XUtil::format("ERROR: Expecting JSON format type, got '%s'.", _PSD.getFormatTypeAsStr().c_str());
    throw std::runtime_error(errMsg);
  }

  std::string sJSONFileName = _PSD.getFile();

  std::fstream fs;
  fs.open(sJSONFileName, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + sJSONFileName;
    throw std::runtime_error(errMsg);
  }

  //  Add a new element to the collection and parse the jason file
  XUtil::TRACE("Reading JSON File: '" + sJSONFileName + '"');
  boost::property_tree::ptree pt;
  try {
    boost::property_tree::read_json(fs, pt);
  } catch (boost::property_tree::json_parser_error &e) {
    std::string errMsg = XUtil::format("ERROR: Parsing the file '%s' on line %d: %s", sJSONFileName.c_str(), e.line(), e.message().c_str());
    throw std::runtime_error(errMsg);
  }
  
  XUtil::TRACE("Examining the property tree from the JSON's file: '" + sJSONFileName + "'");
  XUtil::TRACE("Property Tree: Root");
  XUtil::TRACE_PrintTree("Root", pt);

  for (boost::property_tree::ptree::iterator ptSection = pt.begin(); ptSection != pt.end(); ++ptSection) {
    const std::string & sectionName = ptSection->first;
    if (sectionName == "schema_version") {
      XUtil::TRACE("Skipping: '" + sectionName + "'");
      continue;
    }

    XUtil::TRACE("Processing: '" + sectionName + "'");

    enum axlf_section_kind eKind;
    if (Section::getKindOfJSON(sectionName, eKind) == false) {
      std::string errMsg = XUtil::format("ERROR: Unknown JSON section '%s' in file: %s", sectionName.c_str(), sJSONFileName.c_str());
      throw std::runtime_error(errMsg);
    }

    Section *pSection = findSection(eKind);

    if (pSection == nullptr) {
      Section *pTempSection = Section::createSectionObjectOfKind(eKind);

      // Add DTC exception. Only for 2019.1
      if (eKind == PARTITION_METADATA) {
        pSection = Section::createSectionObjectOfKind(eKind);
        addSection(pSection);
      } else {
        std::string errMsg = XUtil::format("ERROR: Section '%s' doesn't exists for JSON key '%s'.  Must have an existing section in order to append.", pTempSection->getSectionKindAsString().c_str(), sectionName.c_str());
        throw std::runtime_error(errMsg);
      }
    }

    boost::property_tree::ptree ptPayload;
    pSection->getPayload(ptPayload);

    pSection->appendToSectionMetadata(ptSection->second, ptPayload);

    pSection->purgeBuffers();
    pSection->readJSONSectionImage(ptPayload);


    XUtil::TRACE(XUtil::format("Section '%s' (%d) successfully appended to.", pSection->getSectionKindAsString().c_str(), pSection->getSectionKind()));
    std::cout << std::endl << XUtil::format("Section: '%s'(%d) was successfully appended to.\nFormat : %s\nFile   : '%s'", 
                                          pSection->getSectionKindAsString().c_str(), 
                                          pSection->getSectionKind(),
                                          _PSD.getFormatTypeAsStr().c_str(), sectionName.c_str()).c_str() << std::endl;
  }
}

void 
XclBin::dumpSubSection(ParameterSectionData &_PSD)
{
  XUtil::TRACE("Dump Sub-Section");

  // See if there is a subsection to add
  std::string sSubSection = _PSD.getSubSectionName();
  if (sSubSection.empty()) {
    std::string errMsg = XUtil::format("ERROR: No subsection specified: '%s'", _PSD.getOriginalFormattedString().c_str());
    throw std::runtime_error(errMsg);
  }

  // Get the section kind
  enum axlf_section_kind eKind;
  if (Section::translateSectionKindStrToKind(_PSD.getSectionName(), eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name for the command: %s", _PSD.getSectionName().c_str(), _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  // See if the section support sub-sections
  if (Section::supportsSubSections(eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name.", _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  // Determine if the section exists
  Section *pSection = findSection(eKind, _PSD.getSectionIndexName());
  if (pSection == nullptr) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' does not exist.", pSection->getSectionKindAsString().c_str());
    throw std::runtime_error(errMsg);
  }

  // Check to see if the subsection is supported
  if (pSection->supportsSubSection(sSubSection) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' does not support the subsection: '%s'", pSection->getSectionKindAsString().c_str(), sSubSection.c_str());
    throw std::runtime_error(errMsg);
  }

  // Check to see if this subsection exists
  std::ostringstream buffer;
  if (pSection->subSectionExists(_PSD.getSubSectionName()) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' subsection '%s' doesn't exists", pSection->getSectionKindAsString().c_str(), sSubSection.c_str());
    throw std::runtime_error(errMsg);
  }

  // At this point we know we can dump the subsection
  std::string sDumpFileName = _PSD.getFile();
  // Write the xclbin file image
  std::fstream oDumpFile;
  oDumpFile.open(sDumpFileName, std::ifstream::out | std::ifstream::binary);
  if (!oDumpFile.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for writing: " + sDumpFileName;
    throw std::runtime_error(errMsg);
  }

  pSection->dumpSubSection(oDumpFile, sSubSection, _PSD.getFormatType());
  XUtil::TRACE(XUtil::format("Section '%s' (%d) dumped.", pSection->getSectionKindAsString().c_str(), pSection->getSectionKind()));
  std::cout << std::endl << XUtil::format("Section: '%s'(%d) was successfully written.\nFormat: %s\nFile  : '%s'", 
                                          pSection->getSectionKindAsString().c_str(), 
                                          pSection->getSectionKind(),
                                          _PSD.getFormatTypeAsStr().c_str(), sDumpFileName.c_str()).c_str() << std::endl;
}


void 
XclBin::dumpSection(ParameterSectionData &_PSD) 
{
  XUtil::TRACE("Dump Section");

  // See if the user is attempting to dump a sub-section
  if (!_PSD.getSubSectionName().empty()) {
    dumpSubSection(_PSD);
    return;
  }

  enum axlf_section_kind eKind;
  if (Section::translateSectionKindStrToKind(_PSD.getSectionName(), eKind) == false) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' isn't a valid section name.", _PSD.getSectionName().c_str());
    throw std::runtime_error(errMsg);
  }

  const Section *pSection = findSection(eKind);
  if (pSection == nullptr) {
    std::string errMsg = XUtil::format("ERROR: Section '%s' does not exists.", _PSD.getSectionName().c_str());
    throw XUtil::XclBinUtilException(XET_MISSING_SECTION, errMsg);
  }

  if (_PSD.getFormatType() == Section::FT_UNKNOWN) {
    std::string errMsg = "ERROR: Unknown format type '" + _PSD.getFormatTypeAsStr() + "' in the dump section option: '" + _PSD.getOriginalFormattedString() + "'";
    throw std::runtime_error(errMsg);
  }

  if (_PSD.getFormatType() == Section::FT_UNDEFINED ) {
    std::string errMsg = "ERROR: The format type is missing from the dump section option: '" + _PSD.getOriginalFormattedString() + "'.  Expected: <SECTION>:<FORMAT>:<OUTPUT_FILE>.  See help for more format details.";
    throw std::runtime_error(errMsg);
  }

  if (pSection->doesSupportDumpFormatType(_PSD.getFormatType()) == false) {
    std::string errMsg = XUtil::format("ERROR: The %s section does not support writing to a %s file type.",
                                        pSection->getSectionKindAsString().c_str(),
                                        _PSD.getFormatTypeAsStr().c_str());
    throw std::runtime_error(errMsg);
  }

  std::string sDumpFileName = _PSD.getFile();
  // Write the xclbin file image
  std::fstream oDumpFile;
  oDumpFile.open(sDumpFileName, std::ifstream::out | std::ifstream::binary);
  if (!oDumpFile.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for writing: " + sDumpFileName;
    throw std::runtime_error(errMsg);
  }

  pSection->dumpContents(oDumpFile, _PSD.getFormatType());
  XUtil::TRACE(XUtil::format("Section '%s' (%d) dumped.", pSection->getSectionKindAsString().c_str(), pSection->getSectionKind()));
  std::cout << std::endl << XUtil::format("Section: '%s'(%d) was successfully written.\nFormat: %s\nFile  : '%s'", 
                                          pSection->getSectionKindAsString().c_str(), 
                                          pSection->getSectionKind(),
                                          _PSD.getFormatTypeAsStr().c_str(), sDumpFileName.c_str()).c_str() << std::endl;
}

void 
XclBin::dumpSections(ParameterSectionData &_PSD) 
{
  if (!_PSD.getSectionName().empty()) {
    std::string errMsg = "ERROR: Section given for a wildcard JSON section to dump is not empty.";
    throw std::runtime_error(errMsg);
  }

  if (_PSD.getFormatType() != Section::FT_JSON) {
    std::string errMsg = XUtil::format("ERROR: Expecting JSON format type, got '%s'.", _PSD.getFormatTypeAsStr().c_str());
    throw std::runtime_error(errMsg);
  }

  std::string sDumpFileName = _PSD.getFile();
  // Write the xclbin file image
  std::fstream oDumpFile;
  oDumpFile.open(sDumpFileName, std::ifstream::out | std::ifstream::binary);
  if (!oDumpFile.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for writing: " + sDumpFileName;
    throw std::runtime_error(errMsg);
  }

  switch (_PSD.getFormatType()) {
    case Section::FT_JSON:
      {
        boost::property_tree::ptree pt;
        for (auto pSection : m_sections) {
          std::string sectionName = pSection->getSectionKindAsString();
          std::cout << "Examining: '" + sectionName << std::endl;
          pSection->getPayload(pt);
        }

        boost::property_tree::write_json(oDumpFile, pt, true /*Pretty print*/);
        break;
      }
    case Section::FT_HTML:
    case Section::FT_RAW:
    case Section::FT_TXT:
    case Section::FT_UNDEFINED:
    case Section::FT_UNKNOWN:
    default:
      break;
  }

  std::cout << std::endl << XUtil::format("Successfully wrote all of sections which support the format '%s' to the file: '%s'", 
                                          _PSD.getFormatTypeAsStr().c_str(), sDumpFileName.c_str()).c_str() << std::endl;
}


void 
XclBin::setKeyValue(const std::string & _keyValue)
{
  const std::string& delimiters = ":";      // Our delimiter

  // Working variables
  std::string::size_type pos = 0;
  std::string::size_type lastPos = 0;
  std::vector<std::string> tokens;

  // Parse the string until the entire string has been parsed or 3 tokens have been found
  while((lastPos < _keyValue.length() + 1) && 
        (tokens.size() < 3))
  {
    pos = _keyValue.find_first_of(delimiters, lastPos);

    if ( (pos == std::string::npos) ||
         (tokens.size() == 2) ){
       pos = _keyValue.length();
    }

    std::string token = _keyValue.substr(lastPos, pos-lastPos);
    tokens.push_back(token);
    lastPos = pos + 1;
  }

  if (tokens.size() != 3) {
    std::string errMsg = XUtil::format("ERROR: Expected format [USER | SYS]:<key>:<value> when using adding a key value pair.  Received: %s.", _keyValue.c_str());
    throw std::runtime_error(errMsg);
  }

  boost::to_upper(tokens[0]);
  std::string sDomain = tokens[0];
  std::string sKey = tokens[1];
  std::string sValue = tokens[2];
  
  XUtil::TRACE(XUtil::format("Setting key-value pair \"%s\":  domain:'%s', key:'%s', value:'%s'", 
                             _keyValue.c_str(), sDomain.c_str(), sKey.c_str(), sValue.c_str()));

  if (sDomain == "SYS") {
    if (sKey == "mode") {
      if (sValue == "flat" ) {
        m_xclBinHeader.m_header.m_mode = XCLBIN_FLAT;
      } else if (sValue == "hw_pr") {
        m_xclBinHeader.m_header.m_mode = XCLBIN_PR;
      } else if (sValue == "tandem") {
        m_xclBinHeader.m_header.m_mode = XCLBIN_TANDEM_STAGE2;
      } else if (sValue == "tandem_pr") {
        m_xclBinHeader.m_header.m_mode = XCLBIN_TANDEM_STAGE2_WITH_PR;
      } else if (sValue == "hw_emu") {
        m_xclBinHeader.m_header.m_mode = XCLBIN_HW_EMU;
      } else if (sValue == "sw_emu") {
        m_xclBinHeader.m_header.m_mode = XCLBIN_SW_EMU;
      } else {
        std::string errMsg = XUtil::format("ERROR: Unknown value '%s' for key '%s'. Key-value pair: '%s'.", sValue.c_str(), sKey.c_str(), _keyValue.c_str());
        throw std::runtime_error(errMsg);
      }
      return; // Key processed 
    }

    if (sKey == "FeatureRomTimestamp") {
      m_xclBinHeader.m_header.m_featureRomTimeStamp = XUtil::stringToUInt64(sValue);
      return; // Key processed 
    }

    if (sKey == "FeatureRomUUID") {
      sValue.erase(std::remove(sValue.begin(), sValue.end(), '-'), sValue.end()); // Remove the '-'
      XUtil::hexStringToBinaryBuffer(sValue, (unsigned char*)&m_xclBinHeader.m_header.rom_uuid, sizeof(axlf_header::rom_uuid));
      return; // Key processed 
    }

    if (sKey == "PlatformVBNV") {
      XUtil::safeStringCopy((char*)&m_xclBinHeader.m_header.m_platformVBNV, sValue, sizeof(axlf_header::m_platformVBNV));
      return; // Key processed 
    }

    std::string errMsg = XUtil::format("ERROR: Unknown key '%s' for key-value pair '%s'.", sKey.c_str(), _keyValue.c_str());
    throw std::runtime_error(errMsg);
  } 

  if (sDomain == "USER") {
    Section *pSection = findSection(KEYVALUE_METADATA);
    if (pSection == nullptr) {
      pSection = Section::createSectionObjectOfKind(KEYVALUE_METADATA);
      addSection(pSection);
    }

    boost::property_tree::ptree ptKeyValueMetadata;
    pSection->getPayload(ptKeyValueMetadata);

    XUtil::TRACE_PrintTree("KEYVALUE:", ptKeyValueMetadata);
    boost::property_tree::ptree ptKeyValues = ptKeyValueMetadata.get_child("keyvalue_metadata");
    std::vector<boost::property_tree::ptree> keyValues = as_vector<boost::property_tree::ptree>(ptKeyValues, "key_values");

    // Update existing key
    bool bKeyFound = false;
    for (auto &keyvalue : keyValues) {
      if (keyvalue.get<std::string>("key") == sKey) {
         keyvalue.put("value", sValue);
         bKeyFound = true;
         std::cout << "Updating key '" + sKey + "' to '" + sValue + "'" << std::endl;
         break;
      }
    }

    // Need to create a new key
    if (bKeyFound == false) {
      boost::property_tree::ptree keyValue;
      keyValue.put("key", sKey);
      keyValue.put("value", sValue);
      keyValues.push_back(keyValue);
      std::cout << "Creating new key '" + sKey + "' with the value '" + sValue + "'" << std::endl;
    }

    // Now create a new tree to add back into the section
    boost::property_tree::ptree ptKeyValuesNew;
    for (auto keyvalue : keyValues) {
      ptKeyValuesNew.push_back(std::make_pair("", keyvalue));
    }

    boost::property_tree::ptree ptKeyValueMetadataNew;
    ptKeyValueMetadataNew.add_child("key_values", ptKeyValuesNew);

    boost::property_tree::ptree pt;
    pt.add_child("keyvalue_metadata", ptKeyValueMetadataNew);

    XUtil::TRACE_PrintTree("Final KeyValue",pt);
    pSection->readJSONSectionImage(pt);
    return;
  }

  std::string errMsg = XUtil::format("ERROR: Unknown key domain for key-value pair '%s'.  Expected either 'USER' or 'SYS'.", sDomain.c_str());
  throw std::runtime_error(errMsg);
}

void 
XclBin::removeKey(const std::string & _sKey)
{

  XUtil::TRACE(XUtil::format("Removing User Key: '%s'", _sKey.c_str()));

  Section *pSection = findSection(KEYVALUE_METADATA);
  if (pSection == nullptr) {
    std::string errMsg = XUtil::format("ERROR: Key '%s' not found.", _sKey.c_str());
    throw std::runtime_error(errMsg);
  }

  boost::property_tree::ptree ptKeyValueMetadata;
   pSection->getPayload(ptKeyValueMetadata);

   XUtil::TRACE_PrintTree("KEYVALUE:", ptKeyValueMetadata);
   boost::property_tree::ptree ptKeyValues = ptKeyValueMetadata.get_child("keyvalue_metadata");
   std::vector<boost::property_tree::ptree> keyValues = as_vector<boost::property_tree::ptree>(ptKeyValues, "key_values");

   // Update existing key
   bool bKeyFound = false;
   for (unsigned int index = 0; index < keyValues.size(); ++index) {
      if (keyValues[index].get<std::string>("key") == _sKey) {
         bKeyFound = true;
         std::cout << "Removing key '" + _sKey + "'" << std::endl;
         keyValues.erase(keyValues.begin() + index);
         break;
      }
    }

   if (bKeyFound == false) {
     std::string errMsg = XUtil::format("ERROR: Key '%s' not found.", _sKey.c_str());
     throw std::runtime_error(errMsg);
   }

   // Now create a new tree to add back into the section
   boost::property_tree::ptree ptKeyValuesNew;
   for (auto keyvalue : keyValues) {
     ptKeyValuesNew.push_back(std::make_pair("", keyvalue));
   }

   boost::property_tree::ptree ptKeyValueMetadataNew;
   ptKeyValueMetadataNew.add_child("key_values", ptKeyValuesNew);

   boost::property_tree::ptree pt;
   pt.add_child("keyvalue_metadata", ptKeyValueMetadataNew);

   XUtil::TRACE_PrintTree("Final KeyValue",pt);
   pSection->readJSONSectionImage(pt);
   return;
}



void
XclBin::reportInfo(std::ostream &_ostream, const std::string & _sInputFile, bool _bVerbose) const
{
  FormattedOutput::reportInfo(_ostream, _sInputFile, m_xclBinHeader, m_sections, _bVerbose);
}




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

#include "FDTProperty.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

#include "DTCStringsBlock.h"
#include <boost/tuple/tuple.hpp>
#include <boost/tokenizer.hpp>


#include <limits.h>
#include <stdint.h>

#ifdef _WIN32
  #pragma comment(lib, "wsock32.lib")
  #include <winsock2.h>
#else
  #include <arpa/inet.h>
#endif


typedef struct {
  FDTProperty::DataFormat eDataFormat;
  unsigned int wordSize;
  bool isArray;
  std::string extension;
  std::string prettyName;
} DataFormatElement;

static const std::vector< DataFormatElement > _DataFormatTable = {
  { FDTProperty::DF_au8,     1,  true,  "_au8",  "array uint8_t" },
  { FDTProperty::DF_au16,    2,  true,  "_au16", "array uint16_t" },
  { FDTProperty::DF_au32,    4,  true,  "_au32", "array uint32_t" },
  { FDTProperty::DF_au64,    8,  true,  "_au64", "array uint64_t" },
  { FDTProperty::DF_asz,     1,  true,  "_asz",  "string" },
  { FDTProperty::DF_u16,     2,  false, "_u16",  "uint16_t" },
  { FDTProperty::DF_u32,     4,  false, "_u32",  "uint32_t" },
  { FDTProperty::DF_u64,     8,  false, "_u64",  "uint64_t" },
  { FDTProperty::DF_u128,    16, false, "_u128", "uint128_t" },
  { FDTProperty::DF_sz,      1,  false,  "_sz",  "string" },
  { FDTProperty::DF_unknown, 1,  true,  "", "Unknown (default array uint8_t)" }
};

FDTProperty::DataFormat
FDTProperty::getDataFormat(const std::string & _variableName) const
{
  for (unsigned int index = 0; index < _DataFormatTable.size(); ++index) {
    const std::string & extension = _DataFormatTable[index].extension;
    // To handle the case where the table doesn't have an extension
    if (extension.length() == 0) {
      continue;
    }
  
    // Do we have a match  
    if (hasEnding(_variableName, extension) == true) {
      return _DataFormatTable[index].eDataFormat;
    }
  }

  return DF_unknown;
}

const std::string &
FDTProperty::getDataFormatPrettyName(DataFormat _eDataFormat) const
{
  for (unsigned int index = 0; index < _DataFormatTable.size(); ++index) {
    if (_DataFormatTable[index].eDataFormat == _eDataFormat) {
      return _DataFormatTable[index].prettyName;
    }
  }

  // Should never get here, but just in case we do 
  return getDataFormatPrettyName(DF_unknown);
}

unsigned int 
FDTProperty::getWordLength(DataFormat _eDataFormat) const
{
  for (unsigned int index = 0; index < _DataFormatTable.size(); ++index) {
    if (_DataFormatTable[index].eDataFormat == _eDataFormat) {
      return _DataFormatTable[index].wordSize;
    }
  }

  std::string err = XUtil::format("ERROR: Unknown data format: %d", (unsigned int) _eDataFormat);
  throw std::runtime_error(err);
  return 0;
}

bool
FDTProperty::isDataFormatArray(DataFormat _eDataFormat) const
{
  for (unsigned int index = 0; index < _DataFormatTable.size(); ++index) {
    if (_DataFormatTable[index].eDataFormat == _eDataFormat) {
      return _DataFormatTable[index].isArray;
    }
  }
  return false;
}


FDTProperty::FDTProperty()
  : m_dataLength(0)
  , m_pDataBuffer(nullptr)
  , m_eDataFormat(DF_unknown)
{
  // Empty
}


FDTProperty::~FDTProperty() 
{
  // Delete m_pDataBuffer 
  if (m_pDataBuffer != NULL) {
    delete m_pDataBuffer;
    m_pDataBuffer = NULL;
  }
  m_dataLength = 0;
}


void
FDTProperty::runningBufferCheck(const unsigned int _bytesExamined, const unsigned int _size)
{
  if (_bytesExamined > _size) {
    throw std::runtime_error("ERROR: Bytes examined exceeded size of buffer.");
  }
}


struct FDTLenOffset {
  uint32_t len;
  uint32_t nameoff;
};

FDTProperty::FDTProperty(const char* _pBuffer, 
                         const unsigned int _size, 
                         const DTCStringsBlock & _dtcStringsBlock,
                         unsigned int & _bytesExamined,
                         const PropertyNameFormat & _propertyNameFormat)
  : FDTProperty()
{
  XUtil::TRACE("Extracting FDT Property.");

  // Initialize return variables
  _bytesExamined = 0;

  // Validate the buffer
  if (_pBuffer == NULL ) {
     throw std::runtime_error("ERROR: The given property buffer pointer is NULL.");
  }

  if (_size == 0) {
    throw std::runtime_error("ERROR: The given property size is empty.");
  }

  // Check the header size
  if ( _size < sizeof(FDTLenOffset)) {
    std::string err = XUtil::format("ERROR: The given property buffer's header size (%d bytes) is smaller then its header (%d bytes).", _size, sizeof(FDTLenOffset));
    throw std::runtime_error(err);
  }

  // -- Get the offset, length, and name values --
  unsigned int index = 0;

  const FDTLenOffset *pHdr = (const FDTLenOffset *) &_pBuffer[index];
  index += sizeof(FDTLenOffset);
  runningBufferCheck(index, _size);

  m_name = _dtcStringsBlock.getString(ntohl(pHdr->nameoff));
  m_dataLength = ntohl(pHdr->len);

  // -- Determine the type of data
  DataFormat eDataType = getDataFormat(m_name);
  if (_propertyNameFormat.find(m_name) != _propertyNameFormat.end()) {
    eDataType = _propertyNameFormat.find(m_name)->second;
  }

  const std::string & prettyTypeName = getDataFormatPrettyName(eDataType);

  XUtil::TRACE(XUtil::format("Property Name: '%s', length: %d, type: %s", m_name.c_str(), m_dataLength, prettyTypeName.c_str()).c_str());

  // Get the data (if any)
  if (m_dataLength != 0) {
      m_pDataBuffer = new char[m_dataLength];
      memcpy(m_pDataBuffer, &_pBuffer[index], m_dataLength);
      XUtil::TRACE_BUF("Property Data", m_pDataBuffer, m_dataLength);
  } 

  // Update index
  index += m_dataLength;

  // Align to uint32 byte boundary
  if ((index % 4) != 0) {
    index += 4 - (index % 4);
  }

  _bytesExamined = index;
}

bool 
FDTProperty::hasEnding(std::string const &_sFullString, std::string const & _sEndSubString)
{
  // See if there is room
  if (_sFullString.length() < _sEndSubString.length()) {
    return false;
  }

  // Compare the ending of the string to see if they don't match
  if (_sFullString.compare(_sFullString.length() - _sEndSubString.length(), _sEndSubString.length(), _sEndSubString) != 0 ) {
    return false;
  }

  // If we get this far, then they match
  return true;
}


void
FDTProperty::au8MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: Array of 8 bits");

  const uint8_t * uint8Array = (const uint8_t *) m_pDataBuffer;

  boost::property_tree::ptree ptProperty;
  for (unsigned int index = 0; index < m_dataLength; ++index) {
    boost::property_tree::ptree ptChildArrayElement;
    ptChildArrayElement.put("", XUtil::format("0x%x", uint8Array[index]).c_str());
    ptProperty.push_back(std::make_pair("", ptChildArrayElement));
  }
  _ptTree.add_child(m_name.c_str(), ptProperty);
}

void
FDTProperty::au16MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: Array of 16 bits");

  // Check and make sure that all is good 
  static unsigned int byteBoundary = 2;
  if ((m_dataLength % byteBoundary) != 0) {
    std::string err = XUtil::format("ERROR: Data length (%d) does not end on a 2-byte boundary.", m_dataLength);
    throw std::runtime_error(err);
  }

  unsigned int numElements = m_dataLength / byteBoundary;
  const uint16_t * uint16Array = (const uint16_t *) m_pDataBuffer;

  boost::property_tree::ptree ptProperty;
  for (unsigned int index = 0; index < numElements; ++index) {
    boost::property_tree::ptree ptChildArrayElement;
    ptChildArrayElement.put("", XUtil::format("0x%x", ntohs(uint16Array[index])).c_str());
    ptProperty.push_back(std::make_pair("", ptChildArrayElement));
  }
  _ptTree.add_child(m_name.c_str(), ptProperty);
}

void
FDTProperty::au32MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: Array of 32 bits");

  // Check and make sure that all is good 
  static unsigned int byteBoundary = 4;
  if ((m_dataLength % byteBoundary) != 0) {
    std::string err = XUtil::format("ERROR: Data length (%d) does not end on a 4-byte boundary.", m_dataLength);
    throw std::runtime_error(err);
  }

  unsigned int numElements = m_dataLength / byteBoundary;
  const uint32_t * uint32Array = (const uint32_t *) m_pDataBuffer;

  boost::property_tree::ptree ptProperty;
  for (unsigned int index = 0; index < numElements; ++index) {
    boost::property_tree::ptree ptChildArrayElement;
    ptChildArrayElement.put("", XUtil::format("0x%x", ntohl(uint32Array[index])).c_str());
    ptProperty.push_back(std::make_pair("", ptChildArrayElement));
  }
  _ptTree.add_child(m_name.c_str(), ptProperty);
}

void
FDTProperty::u16MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: 16 bits");

  // Check and make sure that all is good 
  static unsigned int byteBoundary = 2;
  if ((m_dataLength % byteBoundary) != 0) {
    std::string err = XUtil::format("ERROR: Data length (%d) does not end on a 2-byte boundary.", m_dataLength);
    throw std::runtime_error(err);
  }

  const uint16_t uint16Value = ntohs(*((const uint16_t *) m_pDataBuffer));
  _ptTree.put(m_name.c_str(), XUtil::format("0x%x", uint16Value).c_str());
}

void
FDTProperty::u32MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: 32 bits");

  // Check and make sure that all is good 
  if (m_dataLength != sizeof(uint32_t)) {
    std::string err = XUtil::format("ERROR: Data length for a 32-bit word is invalid: Expected: %d, Actual: %d", sizeof(uint32_t), m_dataLength);
    throw std::runtime_error(err);
  }

  const uint32_t uint32Value = ntohl(*((const uint32_t *) m_pDataBuffer));
  _ptTree.put(m_name.c_str(), XUtil::format("0x%x", uint32Value).c_str());
}

void
FDTProperty::u128MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: 128 bits");

  // Check and make sure that all is good 
  static unsigned int expectedSize = 16;
  if (m_dataLength != expectedSize) {
    std::string err = XUtil::format("ERROR: Data length for a 128-bit word is invalid: Expected: %d, Actual: %d", expectedSize, m_dataLength);
    throw std::runtime_error(err);
  }

  std::string s128Hex;

  XUtil::binaryBufferToHexString((const unsigned char *) m_pDataBuffer, m_dataLength, s128Hex);
  _ptTree.put(m_name.c_str(), XUtil::format("0x%s", s128Hex.c_str()).c_str());
}

void
FDTProperty::u64MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: 64 bits");

  // Check and make sure that all is good 
  if (m_dataLength != sizeof(uint64_t)) {
    std::string err = XUtil::format("ERROR: Data length for a 64-bit word is invalid: Expected: %d, Actual: %d", sizeof(uint64_t), m_dataLength);
    throw std::runtime_error(err);
  }

  std::string s64Hex;
  XUtil::binaryBufferToHexString((const unsigned char *) m_pDataBuffer, sizeof(uint64_t), s64Hex);
  _ptTree.put(m_name.c_str(), XUtil::format("0x%s", s64Hex.c_str()).c_str());
}

void
FDTProperty::au64MarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: Array 64 bits");

  // Check and make sure that all is good 
  static unsigned int byteBoundary = 8;
  if ((m_dataLength % byteBoundary) != 0) {
    std::string err = XUtil::format("ERROR: Data length (%d) does not end on a 8-byte boundary.", m_dataLength);
    throw std::runtime_error(err);
  }

  boost::property_tree::ptree ptProperty;
  for (unsigned int index = 0; index < m_dataLength; index += byteBoundary) {
    boost::property_tree::ptree ptChildArrayElement;
    std::string s64Hex;
    XUtil::binaryBufferToHexString((const unsigned char *) &m_pDataBuffer[index], byteBoundary, s64Hex);
    ptChildArrayElement.put("", XUtil::format("0x%s", s64Hex.c_str()).c_str());
    ptProperty.push_back(std::make_pair("", ptChildArrayElement));
  }

  _ptTree.add_child(m_name.c_str(), ptProperty);
}


void
FDTProperty::aszMarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: Array String");

  // Check and make sure that all is good 
  if (m_dataLength == 0) {
    throw std::runtime_error("ERROR: Malformed string.  Missing terminator.");
  }

  if (m_pDataBuffer[m_dataLength - 1] != '\0') {
    throw std::runtime_error("ERROR: Missing string terminator.");
  }

  std::vector<std::string> arrayOfStrings;
  unsigned int index = 0;
  unsigned int lastIndex = 0;
  for (index = 0; index < m_dataLength; ++index) {
    if (m_pDataBuffer[index] == '\0') {
      std::string sString = & m_pDataBuffer[lastIndex];
      lastIndex = index + 1;
      arrayOfStrings.push_back(sString);
    }
  }

  boost::property_tree::ptree ptProperty;
  for (auto sString : arrayOfStrings) {
    boost::property_tree::ptree ptChildArrayElement;
    ptChildArrayElement.put("", sString.c_str());
    ptProperty.push_back(std::make_pair("", ptChildArrayElement));
    XUtil::TRACE(XUtil::format("String: %s", sString.c_str()).c_str());
  }

  _ptTree.add_child(m_name.c_str(), ptProperty);
}

void
FDTProperty::szMarshalToJSON(boost::property_tree::ptree &_ptTree) const
{
  XUtil::TRACE("   Type: String");

  // Check and make sure that all is good 
  if (m_dataLength == 0) {
    throw std::runtime_error("ERROR: Malformed string.  Missing terminator.");
  }

  if (m_pDataBuffer[m_dataLength - 1] != '\0') {
    throw std::runtime_error("ERROR: Missing string terminator.");
  }

  _ptTree.put(m_name.c_str(), m_pDataBuffer);
}



unsigned int
FDTProperty::writeDataWord(DataFormat _eDataFormat,
                           char * _buffer, 
                           const std::string & _sData)
{
  unsigned int bytesWritten = 0;
  XUtil::TRACE(XUtil::format("Storing property: '%s' with value: '%s'", m_name.c_str(), _sData.c_str()));

  switch (_eDataFormat) {
    case DF_sz:
    case DF_asz:
      // Copy the string + the '\0' byte
      memcpy(_buffer, _sData.c_str(), (_sData.size() + 1));
      bytesWritten = _sData.size() + 1;
      break;

    case DF_au8:
      {
        uint64_t dataWord = std::strtoul(_sData.c_str(), NULL, 0);
        if (dataWord > UINT8_MAX) {
          std::string err = XUtil::format("ERROR: Property '%s' data value '%s' exceeds the maximum byte storage space'.", m_name.c_str(), _sData.c_str());
          throw std::runtime_error(err);
        }

        uint8_t * pWord = (uint8_t *) _buffer;
        *pWord = (uint8_t) dataWord;
        bytesWritten = sizeof(uint8_t);
      }
      break;
      
    case DF_au16:
    case DF_u16:
      {
        uint64_t dataWord = std::strtoul(_sData.c_str(), NULL, 0);
        if (dataWord > UINT16_MAX) {
          std::string err = XUtil::format("ERROR: Property '%s' data value '%s' exceeds the maximum uint16_t storage space.", m_name.c_str(), _sData.c_str());
          throw std::runtime_error(err);
        }

        uint16_t * pWord = (uint16_t *) _buffer;
        *pWord = htons((uint16_t) dataWord);

        bytesWritten = sizeof(uint16_t);
      }
      break;

    case DF_au32:
    case DF_u32:
      {
        uint64_t dataWord = std::strtoul(_sData.c_str(), NULL, 0);
        if (dataWord > UINT32_MAX) {
          std::string err = XUtil::format("ERROR: Property '%s' data value '%s' exceeds the maximum uint32_t storage space.", m_name.c_str(), _sData.c_str());
          throw std::runtime_error(err);
        }

        uint32_t * pWord = (uint32_t *) _buffer;
        *pWord = htonl((uint32_t) dataWord);

        bytesWritten = sizeof(uint32_t);
      }
      break;

    case DF_au64:
    case DF_u64:
      {
        uint64_t dataWord = std::strtoul(_sData.c_str(), NULL, 0);
        if (errno == ERANGE) {
          std::string err = XUtil::format("ERROR: Property '%s' data value '%s' exceeds the maximum uint64_t storage space.", m_name.c_str(), _sData.c_str());
          throw std::runtime_error(err);
        }

        uint64_t * pWord = (uint64_t *) _buffer;
#ifdef _WIN32
        *pWord = _byteswap_uint64((uint64_t) dataWord);
#else
        *pWord = __builtin_bswap64((uint64_t) dataWord);
#endif

        bytesWritten = sizeof(uint64_t);
      }
      break;

    case DF_u128:
      {
        // Only support hex values
        if ((_sData.compare(0, 2, "0x") != 0) &&
            (_sData.compare(0, 2, "0X") != 0)) {
          std::string err = XUtil::format("ERROR: Property '%s' data value '%s' must be a hex value (e.g., start with '0x').", m_name.c_str(), _sData.c_str());
          throw std::runtime_error(err);
        }

        // Must be of even length
        if ((_sData.size() % 2) != 0) {
            std::string err = XUtil::format("ERROR: Property '%s' data value '%s' doesn't support nibble length values, must be full byte values.", m_name.c_str(), _sData.c_str());
            throw std::runtime_error(err);
        }

        // Must not be too long
        if (_sData.size() > 34) {
          std::string err = XUtil::format("ERROR: Property '%s' data value '%s' exceeds the maximum uint128_t storage space.", m_name.c_str(), _sData.c_str());
          throw std::runtime_error(err);
        }

        std::string sHex(_sData.c_str() + 2);  // Strip off the 2 two characters

        static const int sizeWord = 16;
        uint8_t dataWord[sizeWord] = {};

        XUtil::hexStringToBinaryBuffer(sHex, &dataWord[0], sizeWord);

        memcpy(_buffer, &dataWord[0], sizeWord);

        bytesWritten = sizeWord;
      }
      break;

    case DF_unknown:
    default:
      {
        std::string err = XUtil::format("ERROR: Unknown data type for property '%s'", m_name.c_str());
        throw std::runtime_error(err);
      }
      break;
  }
  return bytesWritten;
}

void 
FDTProperty::marshalDataFromJSON(boost::property_tree::ptree::const_iterator & _iter,
                                 const PropertyNameFormat & _propertyNameFormat)
{
  // Get the name
  m_name = _iter->first;

  m_eDataFormat = getDataFormat(m_name);

  if (_propertyNameFormat.find(m_name) != _propertyNameFormat.end()) {
    m_eDataFormat = _propertyNameFormat.find(m_name)->second;
  }
  unsigned int wordSizeBytes = getWordLength(m_eDataFormat);
  const boost::property_tree::ptree & ptData = _iter->second;
  unsigned int arraySize = (unsigned int) ptData.size();

  // Make sure that we are not dealing with an array of data for non arrays
  if ((arraySize > 1) && !isDataFormatArray(m_eDataFormat)) {
    std::string err = XUtil::format("ERROR: Array of data found for the variable: '%s'", m_name.c_str());
    throw std::runtime_error(err);
  }

  // Address the non-array values first
  if (isDataFormatArray(m_eDataFormat) == false) {
    std::string sData = ptData.data();  

    if (m_eDataFormat == DF_sz) {
      m_dataLength = (unsigned int) sData.size() + 1; // Add room for the '\0' character
    } else {
      m_dataLength = wordSizeBytes;
    }

    m_pDataBuffer = new char[m_dataLength]();
    writeDataWord(m_eDataFormat, m_pDataBuffer, sData);
    return;
  }

  // Just arrays are remaining
  if (m_eDataFormat == DF_asz) {
    m_dataLength = 0;
    for (auto localIter : ptData) {
      std::string sData = localIter.second.data();  
      m_dataLength += sData.size() + 1;  // Add room for the string and the '\0' character
    }
  } else {
    m_dataLength = wordSizeBytes * arraySize;
  }

  m_pDataBuffer = new char[m_dataLength]();

  int byteIndex = 0;
  for (auto localIter : ptData) {
    std::string sData = localIter.second.data();
    byteIndex += writeDataWord(m_eDataFormat, m_pDataBuffer + byteIndex, sData);
  }
  return;
}


FDTProperty::FDTProperty(boost::property_tree::ptree::const_iterator & _iter, 
                         const PropertyNameFormat & _propertyNameFormat)
  : FDTProperty()
{
  marshalDataFromJSON(_iter, _propertyNameFormat);
}






void 
FDTProperty::marshalToJSON(boost::property_tree::ptree &_ptTree,
                           const PropertyNameFormat & _propertyNameFormat) const
{

  DataFormat eDataFormat = getDataFormat(m_name);

  if (_propertyNameFormat.find(m_name) != _propertyNameFormat.end()) {
    eDataFormat = _propertyNameFormat.find(m_name)->second;
  }

  std::string sTypeName = getDataFormatPrettyName(eDataFormat);
  XUtil::TRACE(XUtil::format("-- Serializing Property: '%s', Type: %s", m_name.c_str(), sTypeName.c_str()));

  boost::property_tree::ptree ptProperty;

  switch (eDataFormat) {
    case DF_au8:  au8MarshalToJSON(_ptTree);  break;
    case DF_au16: au16MarshalToJSON(_ptTree); break;
    case DF_au32: au32MarshalToJSON(_ptTree); break;
    case DF_au64: au64MarshalToJSON(_ptTree); break;

    case DF_u16: u16MarshalToJSON(_ptTree); break;
    case DF_u32: u32MarshalToJSON(_ptTree); break;
    case DF_u64: u64MarshalToJSON(_ptTree); break;
    case DF_u128: u128MarshalToJSON(_ptTree); break;

    case DF_asz: aszMarshalToJSON(_ptTree); break;
    case DF_sz: szMarshalToJSON(_ptTree); break;

    default: au8MarshalToJSON(_ptTree); break;
  }
}


#define FDT_PROP        0x00000003

void 
FDTProperty::marshalToDTC(DTCStringsBlock & _dtcStringsBlock, std::ostream& _buf) const
{
  // Add property keyword
  XUtil::write_htonl(_buf, FDT_PROP);

  // Add length and offset values
  XUtil::write_htonl(_buf, m_dataLength);
  XUtil::write_htonl(_buf, _dtcStringsBlock.addString(m_name));

  // Add data (if any)
  if (m_dataLength != 0) {
    _buf.write(m_pDataBuffer, m_dataLength);
  }

  // Pad if necessary
  XUtil::alignBytes(_buf, sizeof(uint32_t));
}

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

#include "XclBinUtilities.h"

#include "Section.h"                           // TODO: REMOVE SECTION INCLUDE

#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <boost/uuid/uuid.hpp>          // for uuid
#include <boost/uuid/uuid_io.hpp>       // for to_string
#include <boost/property_tree/json_parser.hpp>



#ifdef _WIN32
  #include <winsock2.h>
#else
  #include <arpa/inet.h>
#endif

namespace XUtil = XclBinUtilities;

static bool m_bVerbose = false;

void
XclBinUtilities::setVerbose(bool _bVerbose) {
  m_bVerbose = _bVerbose;
  TRACE("Verbosity enabled");
}

void
XclBinUtilities::TRACE(const std::string& _msg, bool _endl) {
  if (!m_bVerbose)
    return;

  std::cout << "Trace: " << _msg.c_str();

  if (_endl)
    std::cout << std::endl << std::flush;
}


void
XclBinUtilities::TRACE_BUF(const std::string& _msg,
                           const char* _pData,
                           uint64_t _size) {
  if (!m_bVerbose)
    return;

  std::ostringstream buf;
  buf << "Trace: Buffer(" << _msg << ") Size: 0x" << std::hex << _size << std::endl;

  buf << std::hex << std::setfill('0');

  uint64_t address = 0;
  while (address < _size) {
    // We know we have data, create the address entry
    buf << "       " << std::setw(8) << address;

    // Read in 16 bytes (or less) at a time
    int bytesRead;
    unsigned char charBuf[16];

    for (bytesRead = 0; (bytesRead < 16) && (address < _size); ++bytesRead, ++address) {
      charBuf[bytesRead] = _pData[address];
    }

    // Show the hex codes
    for (int i = 0; i < 16; i++) {

      // Create a divider ever 8 bytes
      if (i % 8 == 0)
        buf << " ";

      // If we don't have data then display "nothing"
      if (i < bytesRead) {
        buf << " " << std::setw(2) << (unsigned)charBuf[i];
      } else {
        buf << "   ";
      }
    }

    // Bonus: Show printable characters
    buf << "  ";
    for (int i = 0; i < bytesRead; i++) {
      if ((charBuf[i] > 32) && (charBuf[i] <= 126)) {
        buf << charBuf[i];
      } else {
        buf << ".";
      }
    }

    buf << std::endl;
  }

  std::cout << buf.str() << std::endl;
}



void printTree(const boost::property_tree::ptree& pt, std::ostream& _buf = std::cout, int level = 0);



std::string indent(int _level) {
  std::string sIndent;

  for (int i = 0; i < _level; ++i)
    sIndent += "  ";

  return sIndent;
}

void printTree(const boost::property_tree::ptree& pt, std::ostream& _buf, int level) {
  if (pt.empty()) {
    _buf << "\"" << pt.data() << "\"";
  } else {
    if (level)
      _buf << std::endl;

    _buf << indent(level) << "{" << std::endl;

    for (boost::property_tree::ptree::const_iterator pos = pt.begin(); pos != pt.end();) {
      _buf << indent(level + 1) << "\"" << pos->first << "\": ";

      printTree(pos->second, _buf, level + 1);

      ++pos;

      if (pos != pt.end()) {
        _buf << ",";
      }

      _buf << std::endl;
    }

    _buf << indent(level) << " }";
  }

  if (level == 0)
    _buf << std::endl;
}


void
XclBinUtilities::TRACE_PrintTree(const std::string& _msg,
                                 const boost::property_tree::ptree& _pt) {
  if (!m_bVerbose)
    return;

  std::cout << "Trace: Property Tree (" << _msg << ")" << std::endl;

  std::ostringstream outputBuffer;
  boost::property_tree::write_json(outputBuffer, _pt, true /*Pretty print*/);
  std::cout << outputBuffer.str() << std::endl;
  
//  std::ostringstream buf;
//  printTree(_pt, buf);
//  std::cout << buf.str();
}

void
XclBinUtilities::safeStringCopy(char* _destBuffer,
                                const std::string& _source,
                                unsigned int _bufferSize) {
  // Check the parameters
  if (_destBuffer == nullptr)
    return;

  if (_bufferSize == 0)
    return;

  // Initialize the destination buffer with zeros
  memset(_destBuffer, 0, _bufferSize);

  // Determine how many bytes to copy
  unsigned int bytesToCopy = _bufferSize - 1;

  if (_source.size() < bytesToCopy) {
    bytesToCopy = (unsigned int) _source.length();
  }

  // Copy the string
  memcpy(_destBuffer, _source.c_str(), bytesToCopy);
}

unsigned int
XclBinUtilities::bytesToAlign(uint64_t _offset) {
  unsigned int bytesToAlign = (_offset & 0x7) ? 0x8 - (_offset & 0x7) : 0;

  return bytesToAlign;
}

unsigned int
XclBinUtilities::alignBytes(std::ostream & _buf, unsigned int _byteBoundary)
{
  _buf.seekp(0, std::ios_base::end);
  uint64_t bufSize = (uint64_t) _buf.tellp();
  unsigned int bytesAdded = 0;

  if ((bufSize % _byteBoundary) != 0 ) {
    bytesAdded = _byteBoundary - (bufSize % _byteBoundary);
    for (unsigned int index = 0; index < bytesAdded; ++index) {
      char emptyByte = '\0';
      _buf.write(&emptyByte, sizeof(char));
    }
  }

  return bytesAdded;
}


void
XclBinUtilities::binaryBufferToHexString(const unsigned char* _binBuf,
                                         uint64_t _size,
                                         std::string& _outputString) {
  // Initialize output data
  _outputString.clear();

  // Check the data
  if ((_binBuf == nullptr) || (_size == 0)) {
    return;
  }

  // Convert the binary data to an ascii string representation
  std::ostringstream buf;

  for (unsigned int index = 0; index < _size; ++index) {
    buf << std::hex << std::setw(2) << std::setfill('0') << (unsigned int)_binBuf[index];
  }

  _outputString = buf.str();
}


unsigned char
hex2char(const unsigned char _nibbleChar) {
  unsigned char nibble = _nibbleChar;

  if      (_nibbleChar >= '0' && _nibbleChar <= '9') nibble = _nibbleChar - '0';
  else if (_nibbleChar >= 'a' && _nibbleChar <= 'f') nibble = _nibbleChar - 'a' + 10;
  else if (_nibbleChar >= 'A' && _nibbleChar <= 'F') nibble = _nibbleChar - 'A' + 10;

  return nibble;
}


void
XclBinUtilities::hexStringToBinaryBuffer(const std::string& _inputString,
                                         unsigned char* _destBuf,
                                         unsigned int _bufferSize) {
  // Check the data
  if ((_destBuf == nullptr) || (_bufferSize == 0) || _inputString.empty()) {
    std::string errMsg = "Error: hexStringToBinaryBuffer - Invalid parameters";
    throw std::runtime_error(errMsg);
  }

  if (_inputString.length() != _bufferSize * 2) {
    std::string errMsg = "Error: hexStringToBinaryBuffer - Input string is not the same size as the given buffer";
    XUtil::TRACE(XUtil::format("InputString: %d (%s), BufferSize: %d", _inputString.length(), _inputString.c_str(), _bufferSize));
    throw std::runtime_error(errMsg);
  }

  // Initialize buffer
  // Note: We know that the string is even in length
  unsigned int bufIndex = 0;
  for (unsigned int index = 0;
       index < _inputString.length();
       index += 2, ++bufIndex) {
    _destBuf[bufIndex] = (hex2char(_inputString[index]) << 4) + (hex2char(_inputString[index + 1]));
  }
}

#ifdef _WIN32
uint64_t
XclBinUtilities::stringToUInt64(const std::string& _sInteger) {
  uint64_t value = 0;

  // Is it a hex value
  if ((_sInteger.length() > 2) &&
      (_sInteger[0] == '0') &&
      (_sInteger[1] == 'x')) {
    if (1 == sscanf_s(_sInteger.c_str(), "%" PRIx64 "", &value)) {
      return value;
    }
  } else {
    if (1 == sscanf_s(_sInteger.c_str(), "%" PRId64 "", &value)) {
      return value;
    }
  }

  std::string errMsg = "ERROR: Invalid integer string in JSON file: '" + _sInteger + "'";
  throw std::runtime_error(errMsg);
}
#else
uint64_t
XclBinUtilities::stringToUInt64(const std::string& _sInteger) {
  uint64_t value = 0;

  // Is it a hex value
  if ((_sInteger.length() > 2) &&
      (_sInteger[0] == '0') &&
      (_sInteger[1] == 'x')) {
    if (1 == sscanf(_sInteger.c_str(), "%" PRIx64 "", &value)) {
      return value;
    }
  } else {
    if (1 == sscanf(_sInteger.c_str(), "%" PRId64 "", &value)) {
      return value;
    }
  }

  std::string errMsg = "ERROR: Invalid integer string in JSON file: '" + _sInteger + "'";
  throw std::runtime_error(errMsg);
}
#endif

void
XclBinUtilities::printKinds() {
  std::vector< std::string > kinds;
  Section::getKinds(kinds);
  std::cout << "All supported section names supported by this tool:\n";
  for (auto & kind : kinds) {
    std::cout << "  " << kind << "\n";
  }
}

std::string 
XclBinUtilities::getUUIDAsString( const unsigned char (&_uuid)[16] )
{
  static_assert (sizeof(boost::uuids::uuid) == 16, "Error: UUID size mismatch");

  // Copy the values to the UUID structure
  boost::uuids::uuid uuid;
  memcpy((void *) &uuid, (void *) &_uuid, sizeof(boost::uuids::uuid));

  // Now decode it to a string we can work with
  return boost::uuids::to_string(uuid);
}


bool
XclBinUtilities::findBytesInStream(std::fstream& _istream, const std::string& _searchString, unsigned int& _foundOffset) {
  _foundOffset = 0;

  std::iostream::pos_type savedLocation = _istream.tellg();

  unsigned int stringLength = (unsigned int) _searchString.length();
  unsigned int matchIndex = 0;

  char aChar;
  while (_istream.get(aChar)) {
    ++_foundOffset;
    if (aChar == _searchString[matchIndex++]) {
      if (matchIndex == stringLength) {
        _foundOffset -= stringLength;
        return true;
      }
    } else {
      matchIndex = 0;
    }
  }
  _istream.clear();
  _istream.seekg(savedLocation);

  return false;
}

static
const std::string &getSignatureMagicValue()
{
  // Magic Value: 5349474E-9DFF41C0-8CCB82A7-131CC9F3
  unsigned char magicChar[] = { 0x53, 0x49, 0x47, 0x4E, 
                                             0x9D, 0xFF, 0x41, 0xC0, 
                                             0x8C, 0xCB, 0x82, 0xA7, 
                                             0x13, 0x1C, 0xC9, 0xF3};

  static std::string sMagicString((char *) &magicChar[0], 16);

  return sMagicString;
}

bool 
XclBinUtilities::getSignature(std::fstream& _istream, std::string& _sSignature, 
                              std::string& _sSignedBy, unsigned int & _totalSize)
{
  _istream.seekg(0);
  // Find the signature
  unsigned int signatureOffset;
  if (!XclBinUtilities::findBytesInStream(_istream, getSignatureMagicValue(), signatureOffset)) {
    return false;
  }

  // We have a signature read it in
  XUtil::SignatureHeader signature = {0};

  _istream.seekg(signatureOffset);
  _istream.read((char*)&signature, sizeof(XUtil::SignatureHeader));

  // Get signedBy
  if (signature.signedBySize != 0)
  {
    _istream.seekg(signatureOffset + signature.signedByOffset);
    std::unique_ptr<char> data( new char[ signature.signedBySize ] );
    _istream.read( data.get(), signature.signedBySize );
    _sSignedBy = std::string(data.get(), signature.signedBySize);
  }

  // Get the signature
  if (signature.signatureSize != 0)
  {
    _istream.seekg(signatureOffset + signature.signatureOffset);
    std::unique_ptr<char> data( new char[ signature.signatureSize ] );
    _istream.read( data.get(), signature.signatureSize );
    _sSignature = std::string(data.get(), signature.signatureSize);
  }

  _totalSize = signature.totalSignatureSize;
  return true;
}



void 
XclBinUtilities::reportSignature(const std::string& _sInputFile)
{
  // Open the file for consumption
  XUtil::TRACE("Examining xclbin binary file for a signature: " + _sInputFile);
  std::fstream inputStream;
  inputStream.open(_sInputFile, std::ifstream::in | std::ifstream::binary);
  if (!inputStream.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + _sInputFile;
    throw std::runtime_error(errMsg);
  }

  std::string sSignature;
  std::string sSignedBy;
  unsigned int totalSize;
  if (!XUtil::getSignature(inputStream, sSignature, sSignedBy, totalSize)) {
    std::string errMsg = "ERROR: No signature found in file: " + _sInputFile;
    throw std::runtime_error(errMsg);
  }

  std::cout << sSignature << " " << totalSize << std::endl;
}


void 
XclBinUtilities::removeSignature(const std::string& _sInputFile, const std::string& _sOutputFile)
{
  // Open the file for consumption
  XUtil::TRACE("Examining xclbin binary file for a signature: " + _sInputFile);
  std::fstream inputStream;
  inputStream.open(_sInputFile, std::ifstream::in | std::ifstream::binary);
  if (!inputStream.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + _sInputFile;
    throw std::runtime_error(errMsg);
  }

  // Find the signature
  unsigned int signatureOffset;
  if (!XclBinUtilities::findBytesInStream(inputStream, getSignatureMagicValue(), signatureOffset)) {
    std::string errMsg = "ERROR: No signature found in file: " + _sInputFile;
    throw std::runtime_error(errMsg);
  }

  // Open output file
  std::fstream outputStream;
  outputStream.open(_sOutputFile, std::ifstream::out | std::ifstream::binary);
  if (!outputStream.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for writing: " + _sOutputFile;
    throw std::runtime_error(errMsg);
  }

  // Copy the file contents (minus the signature)
  {
    // copy file  
    unsigned int count = 0;
    inputStream.seekg(0);
    char aChar;
    while (inputStream.get(aChar) ) {
      outputStream << aChar;
      if (++count == signatureOffset) {
        break;
      }
    }
  }

  std::cout << "Signature successfully removed." << std::endl;
  outputStream.close();
}

void
createSignatureBufferImage(std::ostringstream& _buf, const std::string & _sSignature, const std::string & _sSignedBy)
{
  XUtil::SignatureHeader signature = {0};
  std::string magicValue = getSignatureMagicValue();

  // Initialize the structure
  unsigned int runningOffset = sizeof(XUtil::SignatureHeader);
  memcpy(&signature.magicValue, magicValue.c_str(), sizeof(XUtil::SignatureHeader::magicValue));

  signature.signatureOffset = runningOffset;
  signature.signatureSize = (unsigned int) _sSignature.size();
  runningOffset += signature.signatureSize;

  signature.signedByOffset = runningOffset;
  signature.signedBySize = (unsigned int) _sSignedBy.size();
  runningOffset += signature.signedBySize;

  signature.totalSignatureSize = runningOffset;

  // Write out the data
  _buf.write(reinterpret_cast<const char*>(&signature), sizeof(XUtil::SignatureHeader));
  _buf.write(_sSignature.c_str(), _sSignature.size());
  _buf.write(_sSignedBy.c_str(), _sSignedBy.size());
}


void
XclBinUtilities::addSignature(const std::string& _sInputFile, const std::string& _sOutputFile,
                              const std::string& _sSignature, const std::string& _sSignedBy)
{
  // Error checks
  if (_sInputFile.empty()) {
    std::string errMsg = "ERROR: Missing file name to modify from.";
    throw std::runtime_error(errMsg);
  }

  // Open the file for consumption
  XUtil::TRACE("Examining xclbin binary file to determine if there is already a signature added: " + _sInputFile);
  std::fstream inputStream;
  inputStream.open(_sInputFile, std::ifstream::in | std::ifstream::binary);
  if (!inputStream.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + _sInputFile;
    throw std::runtime_error(errMsg);
  }

  // See if there already is a signature, if so do nothing
  unsigned int signatureOffset;
  if (XclBinUtilities::findBytesInStream(inputStream, getSignatureMagicValue(), signatureOffset)) {
    std::string errMsg = "ERROR: The given file already has a signature added. File: " + _sInputFile;
    throw std::runtime_error(errMsg);
  }

  // Open output file
  std::fstream outputStream;
  outputStream.open(_sOutputFile, std::ifstream::out | std::ifstream::binary);
  if (!outputStream.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for writing: " + _sOutputFile;
    throw std::runtime_error(errMsg);
  }

  // Copy the file contents
  {
    // copy file  
    inputStream.seekg(0);
    char aChar;
    while (inputStream.get(aChar)) {
      outputStream << aChar;
    }
  }

  // Tack on the signature
  std::ostringstream buffer;
  createSignatureBufferImage(buffer, _sSignature, _sSignedBy);
  outputStream.write(buffer.str().c_str(), buffer.str().size());

  outputStream.close();
}

void 
XclBinUtilities::write_htonl(std::ostream & _buf, uint32_t _word32)
{
  uint32_t word32 = htonl(_word32);
  _buf.write((char *) &word32, sizeof(uint32_t));
}





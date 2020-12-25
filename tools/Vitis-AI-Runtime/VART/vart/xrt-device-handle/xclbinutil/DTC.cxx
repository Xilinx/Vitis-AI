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

#include "DTC.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

#ifdef _WIN32
  #pragma comment(lib, "wsock32.lib")
  #include <winsock2.h>
#else
  #include <arpa/inet.h>
#endif

DTC::DTC() 
  : m_pTopFDTNode(NULL)
{
  // Empty
}

DTC::DTC(const char* _pBuffer, 
         const unsigned int _size,
         const FDTProperty::PropertyNameFormat & _propertyNameFormat) 
  : DTC()
{
  marshalFromDTCImage(_pBuffer, _size, _propertyNameFormat);
}

DTC::DTC(const boost::property_tree::ptree &_ptDTC,
         const FDTProperty::PropertyNameFormat & _propertyNameFormat)
  : DTC()
{
  marshalFromJSON(_ptDTC, _propertyNameFormat);
}


DTC::~DTC() {
  if ( m_pTopFDTNode != NULL ) {
    delete m_pTopFDTNode;
    m_pTopFDTNode = NULL;
  }
}

// DTC Header
struct fdt_header {
  uint32_t magic;
  uint32_t totalsize;
  uint32_t off_dt_struct;
  uint32_t off_dt_strings;
  uint32_t off_mem_rsvmap;
  uint32_t version;
  uint32_t last_comp_version;
  uint32_t boot_cpuid_phys;
  uint32_t size_dt_strings;
  uint32_t size_dt_struct;
};

void
DTC::marshalFromDTCImage( const char* _pBuffer, 
                          const unsigned int _size, 
                          const FDTProperty::PropertyNameFormat & _propertyNameFormat) 
{
  XUtil::TRACE("Marshalling from DTC Image");

  // -- Validate the buffer --
  if ( _pBuffer == NULL ) {
      throw std::runtime_error("ERROR: The given buffer pointer is NULL.");
  }

  static const uint32_t FDT_HEADER_SIZE = sizeof(fdt_header);

  // Check the header size
  if ( _size < FDT_HEADER_SIZE) {
    std::string err = XUtil::format("ERROR: The given DTC buffer's header size (%d bytes) is smaller then the expected size (%d bytes).", _size, FDT_HEADER_SIZE);
    throw std::runtime_error(err);
  }

  const fdt_header * pHdr = (const fdt_header *) _pBuffer;

  // Look for the magic number
  static const uint32_t MAGIC = 0xd00dfeed;

  if ( ntohl(pHdr->magic) != MAGIC ) {
      std::string err = XUtil::format("ERROR: Missing DTC magic number.  Expected: 0x%x, Found: 0x%x.", MAGIC, ntohl(pHdr->magic));
      throw std::runtime_error(err);
  }

  // Validate Size
  if ( ntohl(pHdr->totalsize) != _size ) {
      std::string err = XUtil::format("ERROR: The expected size (%d bytes) does not match actual (%d bytes)", ntohl(pHdr->totalsize), _size);
      throw std::runtime_error(err);
  }

  // -- Get and validate the strings offset --
  if ( ntohl(pHdr->off_dt_strings) > _size ) {
      std::string err = XUtil::format("ERROR: The string block offset (0x%x) exceeds then image size (0x%x)", ntohl(pHdr->off_dt_strings), _size);
      throw std::runtime_error(err);
  }

  if ( (ntohl(pHdr->off_dt_strings) + ntohl(pHdr->size_dt_strings)) > _size ) {
      std::string err = XUtil::format("ERROR: The string block offset and size (0x%x) exceeds then image size (0x%x)", (ntohl(pHdr->off_dt_strings) + ntohl(pHdr->size_dt_strings)), _size);
      throw std::runtime_error(err);
  }

  // -- Get the strings block --
  const char* pStringBuffer = _pBuffer + ntohl(pHdr->off_dt_strings);
  unsigned int stringBufferSize = ntohl(pHdr->size_dt_strings);
  m_DTCStringsBlock.parseDTCStringsBlock(pStringBuffer, stringBufferSize);

  // -- Get the validate the structure nodes offset --
  if ( ntohl(pHdr->off_dt_struct) > _size ) {
      std::string err = XUtil::format("ERROR: The structure block offset (0x%x) exceeds then image size (0x%x)", ntohl(pHdr->off_dt_struct), _size);
      throw std::runtime_error(err);
  }

  if ( (ntohl(pHdr->off_dt_struct) + ntohl(pHdr->size_dt_struct)) > _size ) {
      std::string err = XUtil::format("ERROR: The structure block offset and size (0x%x) exceeds then image size (0x%x)", (ntohl(pHdr->off_dt_struct) + ntohl(pHdr->size_dt_struct)), _size);
      throw std::runtime_error(err);
  }

  // -- Get the top FDT Note (which will include the node tree) --
  const char* pStructureBuffer = _pBuffer + ntohl(pHdr->off_dt_struct);
  unsigned int structureBufferSize = ntohl(pHdr->size_dt_struct);
  m_pTopFDTNode = FDTNode::marshalFromDTC(pStructureBuffer, structureBufferSize, m_DTCStringsBlock, _propertyNameFormat);

  XUtil::TRACE("Marshalling complete");
}


void 
DTC::marshalToJSON(boost::property_tree::ptree &_dtcTree,
                   const FDTProperty::PropertyNameFormat & _propertyNameFormat) const
{
  XUtil::TRACE("");

  if ( m_pTopFDTNode == nullptr ) {
     throw std::runtime_error("ERROR: There are no structure nodes in this design.");
  }

  m_pTopFDTNode->marshalToJSON(_dtcTree, _propertyNameFormat);
}


void 
DTC::marshalFromJSON( const boost::property_tree::ptree &_ptDTC,
                      const FDTProperty::PropertyNameFormat & _propertyNameFormat)
{
    XUtil::TRACE("Marshalling from JSON Image");

    m_pTopFDTNode = FDTNode::marshalFromJSON(_ptDTC, _propertyNameFormat);
}


#define FDT_END         0x00000009
#define FDT_MAGIC       0xd00dfeed

void 
DTC::marshalToDTC(std::ostringstream& _buf) const
{
  XUtil::TRACE("");

  if ( m_pTopFDTNode == NULL ) {
     throw std::runtime_error("ERROR: There are no structure nodes in this design.");
  }

  unsigned int runningOffset = 0;

  // -- Create header --
  fdt_header header = {0};
  header.magic = htonl(FDT_MAGIC);
  header.version = htonl(0x11);
  header.boot_cpuid_phys = htonl(0);
  runningOffset += sizeof(fdt_header);
 
  // -- Create dummy "memory block" --
  std::ostringstream memoryBlock;
  static unsigned int FDTReservedEntrySize = 16;
  for (unsigned index = 0; index < FDTReservedEntrySize; ++index) {
    char emptyByte = '\0';
    memoryBlock.write(&emptyByte, sizeof(char));
  }
  std::string sMemoryBlock = memoryBlock.str();
  header.off_mem_rsvmap = htonl(runningOffset);
  runningOffset += (unsigned int) sMemoryBlock.size();

  // -- Create structure node image --
  std::ostringstream structureNodesBuf;
  DTCStringsBlock stringsBlock;
  m_pTopFDTNode->marshalToDTC(stringsBlock, structureNodesBuf);
  XUtil::write_htonl(structureNodesBuf, FDT_END);
  std::string sStructureNodes = structureNodesBuf.str();
  header.off_dt_struct = htonl(runningOffset);
  header.size_dt_struct = htonl((unsigned long) sStructureNodes.size());
  runningOffset += (unsigned int) sStructureNodes.size();

  // -- Create strings block --
  std::ostringstream stringBlock;
  stringsBlock.marshalToDTC(stringBlock);
  std::string sStringBlock = stringBlock.str();
  header.off_dt_strings = htonl(runningOffset);
  header.size_dt_strings = htonl((unsigned long) sStringBlock.size());
  runningOffset += (unsigned int) sStringBlock.size();

  // Complete the header 
  header.totalsize = htonl(runningOffset);

  // -- Put it all together --
  _buf.write((char *) &header, sizeof(fdt_header));
  _buf.write(sMemoryBlock.c_str(), sMemoryBlock.size());
  _buf.write(sStructureNodes.c_str(), sStructureNodes.size());
  _buf.write(sStringBlock.c_str(), sStringBlock.size());
}

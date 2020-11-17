/**
 * Copyright (C) 2016-2017 Xilinx, Inc
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

// ============================================================================
// COPYRIGHT NOTICE
// (c) Copyright 2017 Xilinx, Inc. All rights reserved.
//
// File Name: xclbindata.h
// ============================================================================

#ifndef __XCLBINDATA_H_
#define __XCLBINDATA_H_

#include "xclbin.h"

#include <map>
#include <vector>
#include <string.h>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <iostream>

// Summary:
//    The methods of this class can be used to read to/write to /dump the context of an xclbin1 files

class XclBinData 
{
  public:
    struct SchemaVersion {
      unsigned int major;
      unsigned int minor;
      unsigned int patch;
    };

  private:
    enum FileMode {
      FM_READ = 0,
      FM_WRITE = 1,
      FM_UNINITIALIZED = 2
    };

  public:
    XclBinData();
    ~XclBinData();

  public:
    void initWrite( const std::string &_file, int _numSections );
    bool writeSectionData( const char* data, size_t size );
    void finishWrite();

    bool initRead( const char* file );
    bool report();
    bool extractBinaryHeader( const char* file, const char* name );
    bool extractAll( const char* name );

    void parseJSONFiles(const std::vector< std::string > & _files);
    unsigned int getJSONBufferSegmentCount();
    void enableTrace() { m_trace = true; };
    void createBinaryImages();
    void createMCSSegmentBuffer(const std::vector< std::pair< std::string, enum MCS_TYPE> > & _mcs);
    void createBMCSegmentBuffer(const std::vector< std::string > & _mps);
 
  private:
    bool extractSectionData( int sectionNum, const char* name );
    void TRACE(const std::string &_msg, bool _endl = true);
    void TRACE_PrintTree(const std::string &_msg, boost::property_tree::ptree &_pt);
    void addPTreeSchemaVersion( boost::property_tree::ptree &_pt, SchemaVersion const &_schemaVersion );
    void getSchemaVersion(boost::property_tree::ptree &_pt, SchemaVersion &_schemaVersion);
    void TRACE_BUF(const std::string &_msg, const char * _pData, unsigned long _size);
    enum MEM_TYPE getMemType( std::string &_sMemType ) const;
    const std::string getMemTypeStr(enum MEM_TYPE _memType) const;
    const std::string getMCSTypeStr(enum MCS_TYPE _mcsType) const;
    enum IP_TYPE getIPType( std::string &_sIPType ) const;
    enum DEBUG_IP_TYPE getDebugIPType( std::string &_sDebugIPType ) const;
    void createMemTopologyBinaryImage( boost::property_tree::ptree &_pt, std::ostringstream &_buf);
    void createConnectivityBinaryImage( boost::property_tree::ptree &_pt, std::ostringstream &_buf);
    void createIPLayoutBinaryImage( boost::property_tree::ptree &_pt, std::ostringstream &_buf);
    void createDebugIPLayoutBinaryImage( boost::property_tree::ptree &_pt, std::ostringstream &_buf);
    void extractMemTopologyData( char * _pDataSegment, unsigned int _segmentSize, boost::property_tree::ptree & _ptree);
    void extractConnectivityData( char * _pDataSegment, unsigned int _segmentSize,boost::property_tree::ptree & _ptree);
    const std::string getIPTypeStr(enum IP_TYPE _ipType) const;
    void extractIPLayoutData( char * _pDataSegment, unsigned int _segmentSize, boost::property_tree::ptree & _ptree); 
    const std::string getDebugIPTypeStr(enum DEBUG_IP_TYPE _debugIpType) const;
    void extractDebugIPLayoutData( char * _pDataSegment, unsigned int _segmentSize, boost::property_tree::ptree & _ptree); 
    enum CLOCK_TYPE getClockType( std::string &_sClockFreqType ) const;
    void createClockFreqTopologyBinaryImage( boost::property_tree::ptree &_pt, std::ostringstream &_buf);
    const std::string getClockTypeStr(enum CLOCK_TYPE _clockFreqType) const;
    void extractClockFreqTopology( char * _pDataSegment, unsigned int _segmentSize,boost::property_tree::ptree & _ptree);
    void extractAndWriteMCSImages( char * _pDataSegment, unsigned int _segmentSize);
    void extractAndWriteBMCImages( char * _pDataSegment, unsigned int _segmentSize);

  private:
    std::string kindToString( axlf_section_kind kind );
    bool reportHead();
    bool reportHeader();
    bool reportSectionHeader( int sectionNum );
    bool reportSectionHeaders(); 

    bool readHead( axlf & head );
    bool readHeader( axlf_section_header & header, int sectionNum );
    std::string getUUIDAsString( const unsigned char (&_uuid)[16] );

  public:
    axlf& getHead() { return m_xclBinHead; }
    void addSection( axlf_section_header& sh, const char* data, size_t size );

  private:
    void align(); // Will align m_xclbinFile to 8 byte boundary.

  private:
    FileMode m_mode;
    unsigned int m_numSections;
    bool m_trace;
    std::fstream m_xclbinFile;
    axlf m_xclBinHead;
    std::vector< axlf_section_header > m_sections;
    std::map< /*axlf_section_kind*/ uint32_t, int > m_sectionCounts;

  private: 
    boost::property_tree::ptree m_ptree_extract;

  public:
    std::map<std::string, boost::property_tree::ptree> m_ptree_segments;

    std::ostringstream m_memTopologyBuf;
    std::ostringstream m_connectivityBuf;
    std::ostringstream m_ipLayoutBuf;
    std::ostringstream m_debugIpLayoutBuf;
    std::ostringstream m_clockFreqTopologyBuf;

    std::ostringstream m_mcsBuf;
    std::ostringstream m_bmcBuf;

    SchemaVersion m_schemaVersion;

    typedef enum {
      BIST_MEM_TOPOLOGY = 0,
      BIST_CONNECTIVITY,
      BIST_IP_LAYOUT,
      BIST_DEBUG_IP_LAYOUT
    } BinaryImageSegType;
};

#endif // __XCLBINDATA_H_




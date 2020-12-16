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

#ifndef __SectionMCS_h_
#define __SectionMCS_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// --------------- C L A S S :   S e c t i o n M C S -------------------------

class SectionMCS : public Section {
 public:
  virtual bool supportsSubSection(const std::string &_sSubSectionName) const;
  virtual bool subSectionExists(const std::string &_sSubSectionName) const;

 protected:
  virtual void getSubPayload(char* _pDataSection, unsigned int _sectionSize, std::ostringstream &_buf, const std::string &_sSubSectionName, enum Section::FormatType _eFormatType) const;
  virtual void marshalToJSON(char* _buffer, unsigned int _pDataSegment, boost::property_tree::ptree& _ptree) const;
  virtual void readSubPayload(const char* _pOrigDataSection, unsigned int _origSectionSize,  std::fstream& _istream, const std::string & _sSubSection, enum Section::FormatType _eFormatType, std::ostringstream &_buffer) const;
  virtual void writeSubPayload(const std::string & _sSubSectionName, FormatType _eFormatType, std::fstream&  _oStream) const;

 protected:
  enum MCS_TYPE getMCSTypeEnum(const std::string & _sSubSectionType) const;
  const std::string getMCSTypeStr(enum MCS_TYPE _mcsType) const;

  typedef std::pair< enum MCS_TYPE, std::ostringstream *> mcsBufferPair;
  void extractBuffers(const char* _pDataSection, unsigned int _sectionSize, std::vector<mcsBufferPair> &_mcsBuffers) const;
  void buildBuffer(const std::vector<mcsBufferPair> &_mcsBuffers, std::ostringstream &_buffer) const;

 public:
  SectionMCS();
  virtual ~SectionMCS();

 private:
  // Purposefully private and undefined ctors...
  SectionMCS(const SectionMCS& obj);
  SectionMCS& operator=(const SectionMCS& obj);

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(MCS, "MCS", "", true, false, boost::factory<SectionMCS*>()); }
  } _initializer;
};

#endif

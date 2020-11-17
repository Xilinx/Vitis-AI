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

#ifndef __SectionBMC_h_
#define __SectionBMC_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// --------------- C L A S S :   S e c t i o n B M C -------------------------

class SectionBMC : public Section {
 public:
  SectionBMC();
  virtual ~SectionBMC();

public:
  enum SubSection{
    SS_UNKNOWN,
    SS_FW,
    SS_METADATA
  };

 public:
  virtual bool doesSupportAddFormatType(FormatType _eFormatType) const;
  virtual bool supportsSubSection(const std::string &_sSubSectionName) const;
  virtual bool subSectionExists(const std::string &_sSubSectionName) const;

 protected:
  virtual void readSubPayload(const char* _pOrigDataSection, unsigned int _origSectionSize,  std::fstream& _istream, const std::string & _sSubSection, enum Section::FormatType _eFormatType, std::ostringstream &_buffer) const;

 protected:
   enum SubSection getSubSectionEnum(const std::string _sSubSectionName) const;
   void copyBufferUpdateMetadata(const char* _pOrigDataSection, unsigned int _origSectionSize,  std::fstream& _istream, std::ostringstream &_buffer) const;
   void createDefaultFWImage(std::fstream & _istream, std::ostringstream &_buffer) const;
   virtual void writeSubPayload(const std::string & _sSubSectionName, FormatType _eFormatType, std::fstream&  _oStream) const;
   void writeFWImage(std::ostream& _oStream) const;
   void writeMetadata(std::ostream& _oStream) const;

 private:
  // Purposefully private and undefined ctors...
  SectionBMC(const SectionBMC& obj);
  SectionBMC& operator=(const SectionBMC& obj);

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(BMC, "BMC", "", true, false, boost::factory<SectionBMC*>()); }
  } _initializer;
};

#endif

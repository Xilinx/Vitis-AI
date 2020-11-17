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

#ifndef __SectionKeyValueMetadata_h_
#define __SectionKeyValueMetadata_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ----- C L A S S :   S e c t i o n K e y V a l u e M e t a d a t a ---------

class SectionKeyValueMetadata : public Section {
 public:
  SectionKeyValueMetadata();
  virtual ~SectionKeyValueMetadata();

 public:
  virtual bool doesSupportAddFormatType(FormatType _eFormatType) const;
  virtual bool doesSupportDumpFormatType(FormatType _eFormatType) const;

 protected:
  virtual void marshalToJSON(char* _pDataSection, unsigned int _sectionSize, boost::property_tree::ptree& _ptree) const;
  virtual void marshalFromJSON(const boost::property_tree::ptree& _ptSection, std::ostringstream& _buf) const;

 private:
  // Purposefully private and undefined ctors...
  SectionKeyValueMetadata(const SectionKeyValueMetadata& obj);
  SectionKeyValueMetadata& operator=(const SectionKeyValueMetadata& obj);

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(KEYVALUE_METADATA, "KEYVALUE_METADATA", "keyvalue_metadata", false, false, boost::factory<SectionKeyValueMetadata*>()); }
  } _initializer;
};

#endif

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

#ifndef __SectionConnectivity_h_
#define __SectionConnectivity_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ------ C L A S S :   S e c t i o n C o n n e c t i v i t y ----------------

class SectionConnectivity : public Section {
 public:
  SectionConnectivity();
  virtual ~SectionConnectivity();

public:
  virtual bool doesSupportAddFormatType(FormatType _eFormatType) const;
  virtual bool doesSupportDumpFormatType(FormatType _eFormatType) const;


 private:
  // Purposefully private and undefined ctors...
  SectionConnectivity(const SectionConnectivity& obj);
  SectionConnectivity& operator=(const SectionConnectivity& obj);

 protected:
  virtual void marshalToJSON(char* _DataSection, unsigned int _sectionSize, boost::property_tree::ptree& _ptree) const;
  virtual void marshalFromJSON(const boost::property_tree::ptree& _ptSection, std::ostringstream& _buf) const;

 protected:

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(CONNECTIVITY, "CONNECTIVITY", "connectivity", false, false, boost::factory<SectionConnectivity*>()); }
  } _initializer;
};

#endif

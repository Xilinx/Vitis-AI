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

#ifndef __SectionMemTopology_h_
#define __SectionMemTopology_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ------- C L A S S :   S e c t i o n M e m T o p o l o g y -----------------

class SectionMemTopology : public Section {
 public:
  SectionMemTopology();
  virtual ~SectionMemTopology();

 public:
  virtual bool doesSupportAddFormatType(FormatType _eFormatType) const;
  virtual bool doesSupportDumpFormatType(FormatType _eFormatType) const;

 protected:
  virtual void marshalToJSON(char* _pDataSection, unsigned int _sectionSize, boost::property_tree::ptree& _ptree) const;
  virtual void marshalFromJSON(const boost::property_tree::ptree& _ptSection, std::ostringstream& _buf) const;

 protected:
  const std::string getMemTypeStr(enum MEM_TYPE _memType) const;
  enum MEM_TYPE getMemType(std::string& _sMemType) const;

 private:
  // Purposefully private and undefined ctors...
  SectionMemTopology(const SectionMemTopology& obj);
  SectionMemTopology& operator=(const SectionMemTopology& obj);

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(MEM_TOPOLOGY, "MEM_TOPOLOGY", "mem_topology", false, false, boost::factory<SectionMemTopology*>()); }
  } _initializer;
};

#endif

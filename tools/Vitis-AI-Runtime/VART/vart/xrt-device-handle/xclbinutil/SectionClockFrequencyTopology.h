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

#ifndef __SectionClockFrequencyTopology_h_
#define __SectionClockFrequencyTopology_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// - C L A S S :   S e c t i o n C l o c k F r e q u e n c y T o p o l o g y -


class SectionClockFrequencyTopology : public Section {
 public:
  SectionClockFrequencyTopology();
  virtual ~SectionClockFrequencyTopology();

 public:
  virtual bool doesSupportAddFormatType(FormatType _eFormatType) const;
  virtual bool doesSupportDumpFormatType(FormatType _eFormatType) const;

 protected:
  virtual void marshalToJSON(char* _pDataSection, unsigned int _sectionSize, boost::property_tree::ptree& _ptree) const;
  virtual void marshalFromJSON(const boost::property_tree::ptree& _ptSection, std::ostringstream& _buf) const;

 protected:
  const std::string getClockTypeStr(enum CLOCK_TYPE _clockType) const;
  enum CLOCK_TYPE getClockType(std::string& _sClockType) const;

 private:
  // Purposefully private and undefined ctors...
  SectionClockFrequencyTopology(const SectionClockFrequencyTopology& obj);
  SectionClockFrequencyTopology& operator=(const SectionClockFrequencyTopology& obj);

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(CLOCK_FREQ_TOPOLOGY, "CLOCK_FREQ_TOPOLOGY", "clock_freq_topology", false, false, boost::factory<SectionClockFrequencyTopology*>()); }
  } _initializer;
};

#endif

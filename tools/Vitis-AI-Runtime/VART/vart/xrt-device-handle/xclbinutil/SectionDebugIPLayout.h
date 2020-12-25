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

#ifndef __SectionDebugIPLayout_h_
#define __SectionDebugIPLayout_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ------ C L A S S :   S e c t i o n D e b u g I P L a y o u t --------------

class SectionDebugIPLayout : public Section {
 private:
  // Purposefully private and undefined ctors...
  SectionDebugIPLayout(const SectionDebugIPLayout& obj);
  SectionDebugIPLayout& operator=(const SectionDebugIPLayout& obj);

 public:
  virtual bool doesSupportAddFormatType(FormatType _eFormatType) const;
  virtual bool doesSupportDumpFormatType(FormatType _eFormatType) const;

 protected:
  virtual void marshalToJSON(char* _pDataSection, unsigned int _sectionSize, boost::property_tree::ptree& _ptree) const;
  virtual void marshalFromJSON(const boost::property_tree::ptree& _ptSection, std::ostringstream& _buf) const;

 protected:
  const std::string getDebugIPTypeStr(enum DEBUG_IP_TYPE _debugIpType) const;
  enum DEBUG_IP_TYPE getDebugIPType(std::string& _sDebugIPType) const;

 public:
  SectionDebugIPLayout();
  virtual ~SectionDebugIPLayout();

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(DEBUG_IP_LAYOUT, "DEBUG_IP_LAYOUT", "debug_ip_layout", false, false, boost::factory<SectionDebugIPLayout*>()); }
  } _initializer;
};

#endif

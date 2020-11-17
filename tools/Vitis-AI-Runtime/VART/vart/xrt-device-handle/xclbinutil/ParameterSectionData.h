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

#ifndef __ParameterSectionData_h_
#define __ParameterSectionData_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include <string>
#include "Section.h"
#include "xclbin.h"

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ------------------- C L A S S :   S e c t i o n ---------------------------

/**
 *    This class represents the base class for a given Section in the xclbin
 *    archive.  
*/

class ParameterSectionData {
 public:
  ParameterSectionData(const std::string &_formattedString);
  virtual ~ParameterSectionData();

 public:
  const std::string &getFile();
  enum Section::FormatType getFormatType();
  const std::string &getFormatTypeAsStr();
  const std::string &getSectionName();
  const std::string &getSubSectionName();
  const std::string &getSectionIndexName();
  enum axlf_section_kind &getSectionKind();
  const std::string &getOriginalFormattedString();

 protected:
   enum Section::FormatType m_formatType;
   std::string m_formatTypeStr;
   std::string m_file;
   std::string m_section;
   std::string m_subSection;
   std::string m_sectionIndex;
   enum axlf_section_kind m_eKind;
   std::string m_originalString;

 protected:
   void transformFormattedString(const std::string _formattedString);

 private:
  ParameterSectionData();
  // Purposefully private and undefined ctors...
  ParameterSectionData(const ParameterSectionData& obj);
  ParameterSectionData& operator=(const ParameterSectionData& obj);

 private:
};

#endif

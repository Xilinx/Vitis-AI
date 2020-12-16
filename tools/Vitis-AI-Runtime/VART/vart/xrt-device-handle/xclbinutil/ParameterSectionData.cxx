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

// ------ I N C L U D E   F I L E S -------------------------------------------
#include "ParameterSectionData.h"

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

ParameterSectionData::ParameterSectionData(const std::string &_formattedString)
  : m_formatType(Section::FT_UNKNOWN)
  , m_formatTypeStr("")
  , m_file("")
  , m_section("")
  , m_subSection("")
  , m_sectionIndex("")
  , m_eKind(BITSTREAM)
  , m_originalString(_formattedString)
{
  transformFormattedString(_formattedString);
}

ParameterSectionData::~ParameterSectionData()
{
  // Empty
}


void 
ParameterSectionData::transformFormattedString(const std::string _formattedString)
// Items being parsed:
// -- Section --
//    Syntax: <section>:<formatType>:<filename>
//   Example: BUILD_METADATA:JSON:MY_FILE.JSON
//
// -- No Section --
//    Syntax: :<formatType>:<filename>
//   Example: :JSON:MY_FILE.JSON

// -- Section & Subsection --
//    Syntax: <section>-<subSection>:<formatType>:<filename>
//   Example: BMC-METADATA:JSON:MY_FILE.JSON
//
// -- Section, Section Element, & Subsection --
//    Syntax: <section>[<Element>]-<subSection>:<formatType>:<filename>
//   Example: SOFT_KERNEL[kernel1]-METADATA:JSON:MY_FILE.JSON
// 
// Note: A file name can contain a colen (e.g., C:\test)
{
  const std::string delimiters = ":";      // Our delimiter

  // Working variables
  std::string::size_type pos = 0;
  std::string::size_type lastPos = 0;
  std::vector<std::string> tokens;

  // Parse the string until the entire string has been parsed or 3 tokens have been found
  while((lastPos < _formattedString.length() + 1) && 
        (tokens.size() < 3))
  {
    pos = _formattedString.find_first_of(delimiters, lastPos);

    if ( (pos == std::string::npos) ||
         (tokens.size() == 2) ){
       pos = _formattedString.length();
    }

    std::string token = _formattedString.substr(lastPos, pos-lastPos);
    tokens.push_back(token);
    lastPos = pos + 1;
  }

  if (tokens.size() != 3) {
    std::string errMsg = XUtil::format("Error: Expected format <section>:<format>:<file> when using adding a section.  Received: %s.", _formattedString.c_str());
    throw std::runtime_error(errMsg);
  }

  m_file = tokens[2];

  m_formatTypeStr = tokens[1];
  m_formatType = Section::getFormatType(tokens[1]);

  // -- Retrieve the section --
  std::string sSection = tokens[0];
  
  if ( sSection.empty() && (m_formatType != Section::FT_JSON)) {
    std::string errMsg = "Error: Empty sections names are only permitted with JSON format files.";
    throw std::runtime_error(errMsg);
  }

  // -- Break down the section--
  if ( !sSection.empty() ) {
    const std::string subDelimiter = "-";      

    // Extract the sub section name (if there is one)
    std::size_t subIndex = sSection.find_last_of(subDelimiter);
    if (subIndex != std::string::npos) {
      m_subSection = sSection.substr(subIndex + 1);
      sSection = sSection.substr(0, subIndex);
    }

    // Extract the section index (if it is there)
    const std::string sectionIndexStartDelimiter = "[";    
    const char sectionIndexEndDelimiter = ']';    
    std::size_t sectionIndex =  sSection.find_first_of(sectionIndexStartDelimiter, 0);

    // Was the start index found?
    if (sectionIndex != std::string::npos) {
      m_section = sSection.substr(0, sectionIndex);
      
      // We need to have an end delimiter
      if (sectionIndexEndDelimiter != sSection.back()) {
        std::string errMsg = XUtil::format("Error: Expected format <section>[<section_index>]:<format>:<file> when using a section index.  Received: %s.", _formattedString.c_str());
        throw std::runtime_error(errMsg);
      }

      // Extract the index name
      sSection.pop_back();  // Remove ']'
      m_sectionIndex = sSection.substr(sectionIndex + 1);

      // Extract the section name
      m_section = sSection.substr(0, sectionIndex);
    } else {
      m_section = sSection;
    }

    if (m_section.empty()) {
      std::string errMsg = XUtil::format("Error: Missing section name. Expected format <section>[<section_index]:<format>:<file> when using a section index.  Received: %s.", _formattedString.c_str());
      throw std::runtime_error(errMsg);
    }

    // Is the section name is valid
    enum axlf_section_kind eKind;
    if (Section::translateSectionKindStrToKind(m_section, eKind) == false) {
      std::string errMsg = XUtil::format("Error: Section '%s' isn't a valid section name.", m_section.c_str());
      throw std::runtime_error(errMsg);
    }

    // Does the section support subsections
    if (!m_subSection.empty() && (Section::supportsSubSections(eKind) == false)) {
      std::string errMsg = XUtil::format("Error: The section '%s' doesn't support subsections (e.g., '%s').", m_section.c_str(), m_subSection.c_str());
      throw std::runtime_error(errMsg);
    }

    // Does the section support section indexes
    if (!m_sectionIndex.empty() && (Section::supportsSectionIndex(eKind) == false)) {
      std::string errMsg = XUtil::format("Error: The section '%s' doesn't support section indexes (e.g., '%s').", m_section.c_str(), m_sectionIndex.c_str());
      throw std::runtime_error(errMsg);
    }
  }
}

const std::string &
ParameterSectionData::getFile()
{
  return m_file;
}

enum Section::FormatType 
ParameterSectionData::getFormatType()
{
  return m_formatType;
}

const std::string &
ParameterSectionData::getSectionName()
{
  return m_section;
}

const std::string &
ParameterSectionData::getSubSectionName()
{
  return m_subSection;
}

const std::string &
ParameterSectionData::getSectionIndexName()
{
  return m_sectionIndex;
}

enum axlf_section_kind &
ParameterSectionData::getSectionKind()
{
  return m_eKind;
}


const std::string &
ParameterSectionData::getFormatTypeAsStr()
{
  return m_formatTypeStr;
}

const std::string &
ParameterSectionData::getOriginalFormattedString()
{
  return m_originalString;
}

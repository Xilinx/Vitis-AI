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

#ifndef __XclBin_h_
#define __XclBin_h_

#include <string>
#include <fstream>
#include <vector>
#include <boost/property_tree/ptree.hpp>

#include "xclbin.h"
#include "ParameterSectionData.h"

class Section;

class XclBin {
 public:
  struct SchemaVersion {
    unsigned int major;
    unsigned int minor;
    unsigned int patch;
  };

 public:
  XclBin();
  virtual ~XclBin();

 public:
  void reportInfo(std::ostream &_ostream, const std::string & _sInputFile, bool _bVerbose) const;
  void printSections(std::ostream &_ostream) const;

  void readXclBinBinary(const std::string &_binaryFileName, bool _bMigrate = false);
  void writeXclBinBinary(const std::string &_binaryFileName, bool _bSkipUUIDInsertion);
  void removeSection(const std::string & _sSectionToRemove);
  void addSection(ParameterSectionData &_PSD);
  void addSections(ParameterSectionData &_PSD);
  void appendSections(ParameterSectionData &_PSD);
  void replaceSection(ParameterSectionData &_PSD);
  void dumpSection(ParameterSectionData &_PSD);
  void dumpSections(ParameterSectionData &_PSD);
  void setKeyValue(const std::string & _keyValue);
  void removeKey(const std::string & _keyValue);

 public:
  Section *findSection(enum axlf_section_kind _eKind, const std::string _indexName = "");
  std::vector<Section*> getSections();
 private:
  void updateHeaderFromSection(Section *_pSection);
  void readXclBinBinaryHeader(std::fstream& _istream);
  void readXclBinBinarySections(std::fstream& _istream);

  void findAndReadMirrorData(std::fstream& _istream, boost::property_tree::ptree& _mirrorData) const;
  void readXclBinaryMirrorImage(std::fstream& _istream, const boost::property_tree::ptree& _mirrorData);

  void writeXclBinBinaryMirrorData(std::fstream& _ostream, const boost::property_tree::ptree& _mirroredData) const;

  void addHeaderMirrorData(boost::property_tree::ptree& _pt_header);

  void addSection(Section* _pSection);
  void addSubSection(ParameterSectionData &_PSD);
  void dumpSubSection(ParameterSectionData &_PSD);

  void removeSection(const Section* _pSection);

  void updateUUID();

  void initializeHeader(axlf &_xclBinHeader);

  // Should be in their own separate class
 private:
  void readXclBinHeader(const boost::property_tree::ptree& _ptHeader, struct axlf& _axlfHeader);
  void readXclBinSection(std::fstream& _istream, const boost::property_tree::ptree& _ptSection);
  void writeXclBinBinaryHeader(std::fstream& _ostream, boost::property_tree::ptree& _mirroredData);
  void writeXclBinBinarySections(std::fstream& _ostream, boost::property_tree::ptree& _mirroredData);


 protected:
  void addPTreeSchemaVersion(boost::property_tree::ptree& _pt, SchemaVersion const& _schemaVersion);
  void getSchemaVersion(boost::property_tree::ptree& _pt, SchemaVersion& _schemaVersion);

 private:
  std::vector<Section*> m_sections;
  axlf m_xclBinHeader;

 protected:
  SchemaVersion m_SchemaVersionMirrorWrite;


 private:
  // Purposefully private and undefined ctors...
  XclBin(const XclBin& obj);
  XclBin& operator=(const XclBin& obj);
};


#endif

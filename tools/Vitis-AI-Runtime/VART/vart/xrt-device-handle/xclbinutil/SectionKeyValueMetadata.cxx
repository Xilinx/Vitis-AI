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

#include "SectionKeyValueMetadata.h"

#include <boost/property_tree/json_parser.hpp>

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

// Static Variables / Classes
SectionKeyValueMetadata::_init SectionKeyValueMetadata::_initializer;

SectionKeyValueMetadata::SectionKeyValueMetadata() {
  // Empty
}

SectionKeyValueMetadata::~SectionKeyValueMetadata() {
  // Empty
}

void 
SectionKeyValueMetadata::marshalToJSON( char* _pDataSection, 
                                        unsigned int _sectionSize, 
                                        boost::property_tree::ptree& _ptree) const
{
    XUtil::TRACE("");
    XUtil::TRACE("Extracting: KEYVALUE_METADATA");
    boost::property_tree::ptree ptKeyValuesMetadata;

    if (_sectionSize == 0) {
       boost::property_tree::ptree ptEmptyKeyvalues;
       ptKeyValuesMetadata.add_child("key_values", ptEmptyKeyvalues);
    } else {
       std::unique_ptr<unsigned char> memBuffer(new unsigned char[_sectionSize + 1]);
       memcpy((char *) memBuffer.get(), _pDataSection, _sectionSize);
       memBuffer.get()[_sectionSize] = '\0';

       XUtil::TRACE_BUF("KEYVALUE_METADATA", (const char *) memBuffer.get(), _sectionSize+1);

       std::stringstream ss;
       ss.write((char*) memBuffer.get(), _sectionSize);

       try {
         boost::property_tree::read_json(ss, ptKeyValuesMetadata);
       } catch (const std::exception & e) {
         std::string msg("ERROR: Bad JSON format detected while marshaling keyvalue metadata (");
         msg += e.what();
         msg += ").";
         throw std::runtime_error(msg);
       }
    }

    _ptree.add_child("keyvalue_metadata", ptKeyValuesMetadata);
}

void 
SectionKeyValueMetadata::marshalFromJSON(const boost::property_tree::ptree& _ptSection, 
                                      std::ostringstream& _buf) const
{
   boost::property_tree::ptree ptKeyValueMetadataBuffer;
   boost::property_tree::ptree ptKeyValuesBuffer;

   XUtil::TRACE("KEYVALUE_METADATA");
   const boost::property_tree::ptree &ptKeyValueMetadata = _ptSection.get_child("keyvalue_metadata");
   const boost::property_tree::ptree &ptKeyValues = ptKeyValueMetadata.get_child("key_values");
   for (const auto& kv : ptKeyValues) {
     boost::property_tree::ptree ptKeyValue = kv.second;
     {
        boost::property_tree::ptree ptKeyValueBuffer;
        ptKeyValueBuffer.put("key", ptKeyValue.get<std::string>("key"));
        ptKeyValueBuffer.put("value", ptKeyValue.get<std::string>("value"));
        ptKeyValuesBuffer.push_back(std::make_pair("", ptKeyValueBuffer));
     }
   }

   ptKeyValueMetadataBuffer.add_child("key_values", ptKeyValuesBuffer);
   XUtil::TRACE_PrintTree("KeyValueMetaData", ptKeyValueMetadataBuffer);

   boost::property_tree::write_json(_buf, ptKeyValueMetadataBuffer, false );
}

bool 
SectionKeyValueMetadata::doesSupportAddFormatType(FormatType _eFormatType) const
{
  if (_eFormatType == FT_JSON) {
    return true;
  }
  return false;
}

bool 
SectionKeyValueMetadata::doesSupportDumpFormatType(FormatType _eFormatType) const
{
    if ((_eFormatType == FT_JSON) ||
        (_eFormatType == FT_HTML))
    {
      return true;
    }

    return false;
}

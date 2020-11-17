/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "./xclbin_info_imp.hpp"

#include <glog/logging.h>
#include <xclbin.h>

#include <boost/property_tree/json_parser.hpp>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/parse_value.hpp>

#include "../../../xrt-device-handle/xclbinutil/Section.h"
#include "../../../xrt-device-handle/xclbinutil/XclBinClass.h"
#include "../../../xrt-device-handle/xclbinutil/XclBinUtilities.h"
#include "./hbm_config.hpp"

DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0")
namespace XUtil = XclBinUtilities;

namespace vart {
namespace dpu {

template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt,
                         boost::property_tree::ptree::key_type const& key) {
  std::vector<T> r;

  boost::property_tree::ptree::const_assoc_iterator it = pt.find(key);

  if (it != pt.not_found()) {
    for (auto& item : pt.get_child(key)) {
      r.push_back(item.second);
    }
  }
  return r;
}

static void get_hbm_from_sessions(
    std::vector<Section*> sections,
    std::unordered_map<std::string, std::pair<std::string, uint64_t>>&
        hbm_address) {
  boost::property_tree::ptree ptMemTopology;
  for (Section* pSection : sections) {
    if (pSection->getSectionKind() == MEM_TOPOLOGY) {
      boost::property_tree::ptree pt;
      pSection->getPayload(pt);
      if (!pt.empty()) {
        ptMemTopology = pt.get_child("mem_topology");
      }
      break;
    }
  }

  if (ptMemTopology.empty()) {
    std::cout << "   No memory configuration data available." << std::endl;
  }

  std::vector<boost::property_tree::ptree> memDatas =
      as_vector<boost::property_tree::ptree>(ptMemTopology, "m_mem_data");
  for (unsigned int index = 0; index < memDatas.size(); ++index) {
    boost::property_tree::ptree& ptMemData = memDatas[index];

    std::string sName = ptMemData.get<std::string>("m_tag");
    std::string sBaseAddress = ptMemData.get<std::string>("m_base_address");
    std::string sAddressSizeKB = ptMemData.get<std::string>("m_sizeKB");
    uint64_t addressSize = XUtil::stringToUInt64(sAddressSizeKB) * 1024;
    if (sName.find("HBM") == std::string::npos) continue;
    hbm_address[sName] = std::make_pair(sBaseAddress, addressSize);
  }
}

static void get_instances_from_sections(
    const std::vector<Section*> sections,
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::string>>&
        instances_info) {
  boost::property_tree::ptree ptMetaData;

  for (Section* pSection : sections) {
    if (pSection->getSectionKind() == BUILD_METADATA) {
      boost::property_tree::ptree pt;
      pSection->getPayload(pt);
      ptMetaData = pt.get_child("build_metadata", pt);
      break;
    }
  }

  if (ptMetaData.empty()) {
    std::cout << "   No kernel metadata available." << std::endl;
    return;
  }

  // Cross reference data
  std::vector<boost::property_tree::ptree> memTopology;
  std::vector<boost::property_tree::ptree> connectivity;
  std::vector<boost::property_tree::ptree> ipLayout;

  for (auto pSection : sections) {
    boost::property_tree::ptree pt;
    if (MEM_TOPOLOGY == pSection->getSectionKind()) {
      pSection->getPayload(pt);
      memTopology = as_vector<boost::property_tree::ptree>(
          pt.get_child("mem_topology"), "m_mem_data");
    } else if (CONNECTIVITY == pSection->getSectionKind()) {
      pSection->getPayload(pt);
      connectivity = as_vector<boost::property_tree::ptree>(
          pt.get_child("connectivity"), "m_connection");
    } else if (IP_LAYOUT == pSection->getSectionKind()) {
      pSection->getPayload(pt);
      ipLayout = as_vector<boost::property_tree::ptree>(
          pt.get_child("ip_layout"), "m_ip_data");
    }
  }

  boost::property_tree::ptree& ptXclBin = ptMetaData.get_child("xclbin");
  std::vector<boost::property_tree::ptree> userRegions =
      as_vector<boost::property_tree::ptree>(ptXclBin, "user_regions");
  for (auto& userRegion : userRegions) {
    std::vector<boost::property_tree::ptree> kernels =
        as_vector<boost::property_tree::ptree>(userRegion, "kernels");
    if (kernels.size() == 0)
      std::cout << "Kernel(s): <None Found>" << std::endl;

    for (auto& ptKernel : kernels) {
      XUtil::TRACE_PrintTree("Kernel", ptKernel);
      std::string sKernel = ptKernel.get<std::string>("name");
      // std::cout << XUtil::format("%s %s", "Kernel:", sKernel.c_str()).c_str()
      //      << std::endl;

      std::vector<boost::property_tree::ptree> ports =
          as_vector<boost::property_tree::ptree>(ptKernel, "ports");
      std::vector<boost::property_tree::ptree> arguments =
          as_vector<boost::property_tree::ptree>(ptKernel, "arguments");
      std::vector<boost::property_tree::ptree> instances =
          as_vector<boost::property_tree::ptree>(ptKernel, "instances");

      // Instance
      for (auto& ptInstance : instances) {
        std::unordered_map<std::string, std::string> instance_info;
        std::string sInstance = ptInstance.get<std::string>("name");
        std::string sKernelInstance = sKernel + ":" + sInstance;

        // Base Address
        {
          std::string sBaseAddress = "--";
          for (auto& ptIPData : ipLayout) {
            if (ptIPData.get<std::string>("m_name") == sKernelInstance) {
              sBaseAddress = ptIPData.get<std::string>("m_base_address");
              break;
            }
          }
        }

        // List the arguments
        for (unsigned int argumentIndex = 0; argumentIndex < arguments.size();
             ++argumentIndex) {
          boost::property_tree::ptree& ptArgument = arguments[argumentIndex];
          std::string sArgument = ptArgument.get<std::string>("name");
          std::string sOffset = ptArgument.get<std::string>("offset");
          std::string sPort = ptArgument.get<std::string>("port");
          if (sPort.find("DPU") == std::string::npos) continue;

          // Find the memory connections
          bool bFoundMemConnection = false;
          for (auto& ptConnection : connectivity) {
            unsigned int ipIndex =
                ptConnection.get<unsigned int>("m_ip_layout_index");

            if (ipIndex >= ipLayout.size()) {
              std::string errMsg = XUtil::format(
                  "ERROR: connectivity section 'm_ip_layout_index' (%d) "
                  "exceeds the number of 'ip_layout' elements (%d).  This is "
                  "usually an indication of curruptions in the xclbin archive.",
                  ipIndex, ipLayout.size());
              throw std::runtime_error(errMsg);
            }

            if (ipLayout[ipIndex].get<std::string>("m_name") ==
                sKernelInstance) {
              if (ptConnection.get<unsigned int>("arg_index") ==
                  argumentIndex) {
                bFoundMemConnection = true;

                unsigned int memIndex =
                    ptConnection.get<unsigned int>("mem_data_index");
                if (memIndex >= memTopology.size()) {
                  std::string errMsg = XUtil::format(
                      "ERROR: connectivity section 'mem_data_index' (%d) "
                      "exceeds the number of 'mem_topology' elements (%d).  "
                      "This is usually an indication of curruptions in the "
                      "xclbin archive.",
                      memIndex, memTopology.size());
                  throw std::runtime_error(errMsg);
                }

                std::string sMemName =
                    memTopology[memIndex].get<std::string>("m_tag");
                std::string sMemType =
                    memTopology[memIndex].get<std::string>("m_type");
                instance_info[sPort] = sMemName;
                if (sPort.find("0") != std::string::npos) break;
              }
            }
          }
          if (!bFoundMemConnection) {
            std::cout
                << XUtil::format("   %-18s <not applicable>", "Memory:").c_str()
                << std::endl;
          }
        }
        instances_info[sInstance] = instance_info;
      }
    }
  }
}

XclbinInfoImp::XclbinInfoImp(const std::string& xclbin_file) {
  xclbin_.readXclBinBinary(xclbin_file);
  sections_ = xclbin_.getSections();
  get_hbm_from_sessions(sections_, hbm_address_);
  get_instances_from_sections(sections_, instances_);
}

XclbinInfoImp::~XclbinInfoImp() {}

void XclbinInfoImp::show_hbm() {
  std::cout << "HBM info: " << std::endl;
  for (auto& hbm : hbm_address_) {
    std::cout
        << XUtil::format("   %-13s %s", "Index:", hbm.first.c_str()).c_str()
        << std::endl;
    std::cout << XUtil::format("   %-13s %s",
                               "Base Address:", hbm.second.first.c_str())
                     .c_str()
              << std::endl;
    std::cout << XUtil::format("   %-13s 0x%lx",
                               "Address Size:", hbm.second.second)
                     .c_str()
              << std::endl;
    std::cout << std::endl;
  }
}

void XclbinInfoImp::show_instances() {
  for (auto& instance : instances_) {
    std::cout << XUtil::format("%-16s %s", "Instance:", instance.first.c_str())
                     .c_str()
              << std::endl;
    for (auto item : instance.second) {
      std::cout
          << XUtil::format("   %-18s %s", "Port:", item.first.c_str()).c_str()
          << std::endl;
      std::cout << XUtil::format("   %-18s %s", "Memory:", item.second.c_str())
                       .c_str()
                << std::endl;
      std::cout << std::endl;
    }
  }
}

static std::string get_suffix(std::string str) {
  auto pos = str.find_last_of('_');
  return str.substr(pos + 1);
}

std::vector<HbmChannelProperty> XclbinInfoImp::HBM_CHANNELS() {
  auto ret = std::vector<HbmChannelProperty>();
  if (ret.empty()) {
    for (auto& instance_info : instances_) {
      for (auto item : instance_info.second) {
        unsigned int core_id =
            (unsigned int)std::stoul(get_suffix(instance_info.first).c_str());
        std::string name = get_suffix(item.first);
        if (name.find("W") == std::string::npos &&
            name.find("I") == std::string::npos) {
          name = "D" + name;
        };
        std::string offset = hbm_address_[item.second].first;
        uint64_t capacity = hbm_address_[item.second].second;
        uint64_t offset2;
        vitis::ai::parse_value(offset, offset2);
        ret.emplace_back(HbmChannelProperty{name,
                                            core_id,
                                            {{
                                                offset2,
                                                capacity,
                                                4ull * 1024ull,
                                            }}});
      }
    }
  }
  return ret;
}

}  // namespace dpu
}  // namespace vart

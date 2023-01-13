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
#include "./xrt_bin_stream.hpp"
#include <UniLog/UniLog.hpp>
#include <fstream>
#include <glog/logging.h>
#include <xclbin.h>
#include <xrt.h>

#include <cstring>
#include <map>
#include <numeric>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_XRT_BIN_STREAM, "0");
namespace xir {

XrtBinStream::XrtBinStream(const std::string filename) {
  init_fd(filename);
  init_top();
  init_uuid();
  init_ip_layout();
  init_mem_topology();
  init_cu_names();
  init_cu_indices();
  guess_lpddr();
}

XrtBinStream::~XrtBinStream() {
}
void XrtBinStream::init_fd(const std::string filename) {
  std::ifstream t(filename, std::ifstream::binary);
  if(!t.good()) {
    LOG(INFO) << "Please check your /etc/vart.conf\n"
              << "Its format should be :\n    firmware: xx\n"
              << "Example:\n"
              << "    firmware: /run/media/mmcblk0p1/dpu.xclbin";
    UNI_LOG_FATAL(VART_OPEN_DEVICE_FAIL)
      << "open(" << filename << ") failed.";
  }
  t.seekg(0, t.end);
  size_t length = t.tellg();
  t.seekg(0, t.beg);
  buffer_.resize(length);
  CHECK(t.read(&buffer_[0], buffer_.size()).good());
  data_ = &buffer_[0];
}
void XrtBinStream::init_top() { top_ = (const axlf*)data_; }
void XrtBinStream::init_uuid() {  //
  memcpy(&uuid_, top_->m_header.uuid, sizeof(xuid_t));
  dsa_ = (const char*)top_->m_header.m_platformVBNV;
}

void XrtBinStream::init_ip_layout() {
  auto ip = xclbin::get_axlf_section(top_, IP_LAYOUT);
  ip_layout_ = (ip_layout*)(((char*)data_) + ip->m_sectionOffset);
}

void XrtBinStream::init_mem_topology() {
  auto topo = xclbin::get_axlf_section(top_, MEM_TOPOLOGY);
  topology_ = (mem_topology*)(((char*)data_) + topo->m_sectionOffset);
}
void XrtBinStream::init_cu_names() {
  for (auto i = 0; i < ip_layout_->m_count; ++i) {
    if (ip_layout_->m_ip_data[i].m_type != IP_KERNEL) continue;
    cu_names_.push_back(std::string((char*)ip_layout_->m_ip_data[i].m_name));
  }
}

void XrtBinStream::init_cu_indices() {
  indices_ = std::vector<size_t>(get_num_of_cu());
  std::iota(indices_.begin(), indices_.end(), 0u);
  std::sort(indices_.begin(), indices_.end(), [this](size_t a, size_t b) {
    return ip_layout_->m_ip_data[a].m_base_address <
           ip_layout_->m_ip_data[b].m_base_address;
  });
}

void XrtBinStream::guess_lpddr() {
  is_lpddr_ = false;
  for (int i = 0; i < topology_->m_count; ++i) {
    if (topology_->m_mem_data[i].m_used &&
        strncmp((char*)topology_->m_mem_data[i].m_tag, "LPDDR", 5) == 0) {
      is_lpddr_ = true;
    }
  }
}

static std::string to_string(const xuid_t x) {
  char buf[sizeof(xuid_t) * 4 + 1];
  char* p = &buf[0];
  for (auto i = 0u; i < sizeof(xuid_t); ++i) {
    sprintf(p, " %02x", x[i]);
    p = p + strlen(p);
  }
  return std::string(buf);
}

void XrtBinStream::dump_layout() const {
  LOG(INFO) << "uuid: " << to_string(uuid_) << "\nDSA: " << dsa_;
  for (auto i = 0; i < ip_layout_->m_count; ++i) {
    if (ip_layout_->m_ip_data[i].m_type != IP_KERNEL) continue;
    LOG(INFO) << "TYPE: " << ip_layout_->m_ip_data[i].m_type << "\n"
              << "cu[" << i << "] = " << ip_layout_->m_ip_data[i].m_name
              << "\n"                                                   //
              << "ip type " << ip_layout_->m_ip_data[i].m_type << "\n"  //
              << "ip interrupt enabled "                                //
              << (ip_layout_->m_ip_data[i].properties & 0x1) << "\n"    //
              << "ip interrupt id "                                     //
              << ((ip_layout_->m_ip_data[i].properties & 0x000000FE) >> 1)
              << "\n"  //
              << "ip interrupt ctrl "
              << ((ip_layout_->m_ip_data[i].properties & 0x0000FF00) >> 16)
              << "\n"  //
              << "cu base addr: " << std::hex << "0x"
              << ip_layout_->m_ip_data[i].m_base_address << "\n"  //
              << "properties : 0x" << ip_layout_->m_ip_data[i].properties
              << "\n"                         //
              << "is_lpddr_ : " << is_lpddr_  //
              << "\n"                         //
        ;
  }
}

void XrtBinStream::dump_mem_topology() const {
  std::ostringstream str;
  for (int i = 0; i < topology_->m_count; ++i) {
    // if (topology_->m_mem_data[i].m_used) {
    str << (((int)topology_->m_mem_data[i].m_used) ? "o" : "*")
        << (char*)topology_->m_mem_data[i].m_tag << " ";
    // }
  }
  LOG(INFO) << "MEM TOPOLOGY: [" << str.str() << "]";
}
void XrtBinStream::burn(int device_id) {
  auto handle = xclOpen(device_id, NULL, XCL_INFO);
  burn(handle);
  xclClose(handle);
}
void XrtBinStream::burn(xclDeviceHandle handle) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_BIN_STREAM))
      << "burning " << to_string(uuid_);
  const xclBin* blob = (const xclBin*)data_;
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  UNI_LOG_CHECK(xclLockDevice(handle) == 0, VART_LOCK_DEVICE_FAIL);
  UNI_LOG_CHECK(xclLoadXclBin(handle, blob) ==0 , VART_LOAD_XCLBIN_FAIL);
}
std::array<unsigned char, sizeof(xuid_t)> XrtBinStream::get_uuid() const {
  auto ret = std::array<unsigned char, sizeof(xuid_t)>();
  memcpy(&ret[0], &uuid_[0], sizeof(uuid_));
  return ret;
}
size_t XrtBinStream::get_num_of_cu() const { return cu_names_.size(); }
std::string XrtBinStream::get_cu(size_t idx) const {
  UNI_LOG_CHECK(idx < indices_.size(), VART_OUT_OF_RANGE) << " current cu index larger than the total size!";
  return cu_names_[indices_[idx]];
}
uint64_t XrtBinStream::get_cu_base_addr(size_t idx) const {
  return ip_layout_->m_ip_data[indices_[idx]].m_base_address;
}
std::string XrtBinStream::get_dsa() const { return dsa_; }
}  // namespace xir

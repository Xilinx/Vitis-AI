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
#include "./xrt_device_handle_butler.hpp"

#include <UniLog/UniLog.hpp>
#include <glog/logging.h>

#include <algorithm>
#include <iostream>
#include <numeric>

#include "butler_fpga_selection_algo.h"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/simple_config.hpp"
#include "vitis/ai/weak.hpp"
#include "xrt_xcl_read.hpp"
DEF_ENV_PARAM(XLNX_ENABLE_XIP, "1");
DEF_ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE, "0");
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string);
DEF_ENV_PARAM_2(XLNX_ENABLE_DEVICES, "ALL", std::string);
namespace {
const std::string get_dpu_xclbin() {
  auto ret = std::string("/usr/lib/dpu.xclbin");
  if (!ENV_PARAM(XLNX_VART_FIRMWARE).empty()) {
    ret = ENV_PARAM(XLNX_VART_FIRMWARE);
    return ret;
  }
  auto config =
      vitis::ai::SimpleConfig::getOrCreateSimpleConfig("/etc/vart.conf");
  if (!config) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "/etc/vart.conf does not exits. use default value "
           "/usr/lib/dpu.xclbin";
    return ret;
  }
  auto has_firmware = (*config).has("firmware");
  if (!has_firmware) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "/etc/vart.conf does not contains firmware: xxx. use default value "
           "/usr/lib/dpu.xclbin";
    return ret;
  }
  ret = (*config)("firmware").as<std::string>();
  return ret;
}

static std::vector<std::string> split_str(const std::string& str,
                                          const char delim = ',') {
  auto list = std::vector<std::string>();
  std::istringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    list.push_back(item);
  }
  return list;
}
static std::vector<size_t> get_device_id_list(const size_t num_of_devices) {
  std::string enable_devices = ENV_PARAM(XLNX_ENABLE_DEVICES);
  if (enable_devices == "ALL") {
    auto device_id_list = std::vector<size_t>(num_of_devices);
    std::iota(device_id_list.begin(), device_id_list.end(), 0);
    return device_id_list;
  }
  auto device_id_list = std::vector<size_t>();
  for (auto d : split_str(enable_devices, ',')) {
    auto device_id = (size_t)std::stoi(d);
    if (device_id < num_of_devices) {
      device_id_list.push_back(device_id);
    }
  }
  return device_id_list;
}

static bool start_with(const std::string& a, const std::string& b) {
  return a.find(b) == 0u;
}

static uint64_t my_get_fingerprint(const std::string& full_cu_name,
                                   xclDeviceHandle handle, size_t ip_index,
                                   uint64_t base) {
  auto env_fingerprint = getenv((full_cu_name + ".fingerprint").c_str());
  if (env_fingerprint) {
    return std::stoul(env_fingerprint);
  }
  uint32_t h_value = 0;
  uint32_t l_value = 0;
  uint64_t ret = 0u;
  size_t size = sizeof(h_value);
  auto cu_offset = 0x1F0;
  auto read_result = xrtXclRead(handle, ip_index, cu_offset, base, &l_value);
  UNI_LOG_CHECK(read_result == 0, VART_XRT_READ_ERROR) << "xrtXclRead has error!";
  cu_offset = cu_offset + sizeof(uint32_t);
  read_result = xrtXclRead(handle, ip_index, cu_offset + sizeof(uint32_t), base,
                           &h_value);
  UNI_LOG_CHECK(read_result == 0, VART_XRT_READ_ERROR) << "xrtXclRead has error!";
  ret = h_value;
  ret = (ret << 32) + l_value;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << full_cu_name << " fingerprint: 0x" << std::hex << ret << std::dec;

  return ret;
}

class FPGASelection : public butler::FPGASelectionAlgo {
 public:
  FPGASelection(const std::string& dsa) : dsa_(dsa){};

  virtual ~FPGASelection() = default;

 private:
  butler::UDFResult execute(butler::FPGASelectionAlgoParam_t param) const;

 private:
  const std::string dsa_;
};

// DSA naming scheme is too complex.
butler::UDFResult FPGASelection::execute(
    butler::FPGASelectionAlgoParam_t param) const {
  butler::UDFResult rtn;
  rtn.valid = true;
  butler::XCLBIN* xclbinPtr = nullptr;

  const std::vector<butler::XCLBIN>* xclbins_ptr = param.xclbins;
  const std::vector<butler::XCLBIN>& xclbins = *xclbins_ptr;
  const butler::System* system = param.system;

  // check for matching xclbin;
  for (auto j = 0u; (j < xclbins.size()) && (xclbinPtr == nullptr); j++) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "expected dsa " << dsa_ << " xclbins[" << j
        << "].dsa = " << xclbins[j].getDSAName();
    if (dsa_.compare(xclbins[j].getDSAName()) == 0) {
      xclbinPtr = const_cast<butler::XCLBIN*>(&(*(xclbins.begin() + j)));
    }
  }
  int index = -1;
  // xclbin exists
  if (xclbinPtr != nullptr) {
    for (int j = 0; j < system->getNumFPGAs(); j++) {
      const auto* fpga = system->getFPGAAtIdx(j);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "j=" << j << " "                          //
          << "DSAName= " << fpga->getDSAName() << " "  //
          << "Used= " << fpga->AreCUsUsed() << " "     //
          ;
    }
    for (int j = 0; j < system->getNumFPGAs(); j++) {
      auto device_id_list = get_device_id_list(system->getNumFPGAs());
      const auto* fpga = system->getFPGAAtIdx(j);
      if (fpga->AreCUsUsed()) continue;
      if (std::find(device_id_list.begin(), device_id_list.end(), (size_t)j) !=
          device_id_list.end()) {
        index = j;
        break;
      }
    }
  }
  if (0 <= index) {
    const butler::xFPGA* fpgaResource = system->getFPGAAtIdx(index);
    rtn.myErrCode = butler::errCode::SUCCESS;
    rtn.FPGAidx = fpgaResource->getIdx();
    rtn.XCLBINidx = xclbinPtr->getIdx();
  }
  return rtn;
}

// {
//   butler::UDFResult rtn;
//   rtn.valid = true;
//   butler::XCLBIN* xclbinPtr = nullptr;

//   const std::vector<butler::XCLBIN>* xclbins_ptr = param.xclbins;
//   const std::vector<butler::XCLBIN>& xclbins = *xclbins_ptr;
//   const butler::System* system = param.system;

//   // check for matching xclbin
//   for (auto j = 0u; (j < xclbins.size()) && (xclbinPtr == nullptr); j++) {
//     LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
//         << "expected dsa " << dsa_ << " xclbins[" << j
//         << "].dsa = " << xclbins[j].getDSAName();
//     if (dsa_.compare(xclbins[j].getDSAName()) == 0) {
//       xclbinPtr = const_cast<butler::XCLBIN*>(&(*(xclbins.begin() + j)));
//     }
//   }
//   int index = -1;
//   // xclbin exists
//   if (xclbinPtr != nullptr) {
//     for (int j = 0; j < system->getNumFPGAs(); j++) {
//       const auto* fpga = system->getFPGAAtIdx(j);
//       LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
//           << "j=" << j << " "                          //
//           << "DSAName= " << fpga->getDSAName() << " "  //
//           << "Used= " << fpga->AreCUsUsed() << " "     //
//           ;
//     }
//     for (int j = 0; j < system->getNumFPGAs(); j++) {
//       const auto* fpga = system->getFPGAAtIdx(j);
//       if (fpga->AreCUsUsed()) continue;
//       // match FPGA dsa
//       if (dsa_.compare(fpga->getDSAName()) == 0) {
//         index = j;
//         break;
//       }
//     }
//   }
//   if (0 <= index) {
//     const butler::xFPGA* fpgaResource = system->getFPGAAtIdx(index);
//     rtn.myErrCode = butler::errCode::SUCCESS;
//     rtn.FPGAidx = fpgaResource->getIdx();
//     rtn.XCLBINidx = xclbinPtr->getIdx();
//   }
//   LOG(WARNING) << "cannot find matched dsa" << param << " ";
//   return rtn;
// }

static std::string to_string(const std::array<unsigned char, 16ul>& x) {
  char buf[sizeof(xuid_t) * 4 + 1];
  char* p = &buf[0];
  for (auto i = 0u; i < sizeof(xuid_t); ++i) {
    sprintf(p, " %02x", x[i]);
    p = p + strlen(p);
  }
  return std::string(buf);
}

XrtDeviceHandleImp::XrtDeviceHandleImp() {
  auto num_of_devices = xclProbe();
  client_ = std::make_unique<butler::ButlerClient>();
  CHECK(client_->Ping() == butler::errCode::SUCCESS)
      << "cannot ping butler server";
  auto filename = get_dpu_xclbin();
  binstream_ = std::make_unique<xir::XrtBinStream>(filename);
  auto uuid = binstream_->get_uuid();
  auto algo = std::make_unique<FPGASelection>(binstream_->get_dsa());

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "open firmware " << filename << "uuid = " << to_string(uuid);

  auto cu_idx = 0u;
  for (auto deviceIndex = 0u; deviceIndex < num_of_devices; ++deviceIndex) {
    auto acquireResult = client_->acquireFPGA(filename, nullptr, algo.get());
    auto alloc = acquireResult.first;
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "acquireResult= (" << alloc.myErrCode << ", " << alloc.valid
        << "), ";
    if ((alloc.myErrCode == butler::errCode::SUCCESS) && (alloc.valid)) {
    } else {
      continue;
    }
    auto Handle = alloc.myHandle;
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "FPGA resource is accquired from Butler: "
        << "Handle = " << Handle;
    auto xcu = acquireResult.second;
    for (auto& xcux : xcu) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "\n"                                                         //
          << "\tIndex: " << cu_idx << " and " << xcux.getCUIdx() << "\n"  //
          << "\tName: " << xcux.getName() << "\n"                         //
          << "\tDSAName: " << xcux.getDSAName() << "\n"                   //
          << "\tDevName: " << xcux.getDevName() << "\n"                   //
          << "\tKernelName: " << xcux.getKernelName() << "\n"             //
          << "\tXCLBIN: " << xcux.getXCLBINPath() << "\n"                 //
          << "\tgetFPGAIdx: " << xcux.getFPGAIdx() << "\n"                //
          << "\tgetBaseAddrFPGAIdx: 0x" << std::hex << xcux.getBaseAddr()
          << std::dec << "\n"                      //
          << "\tused: " << xcux.getUsed() << "\n"  //
          << "\txDev: " << xcux.getXDev() << "\n"  //
          ;
      auto cu_full_name = xcux.getKernelName() + ":" + xcux.getName();
      auto cu_name = xcux.getName();
      auto& x = handles_[cu_full_name];
      x.cu_base_addr = xcux.getBaseAddr();
      x.cu_index = xcux.getCUIdx();
      x.handle = xclOpen(xcux.getFPGAIdx(), NULL, XCL_INFO);
      x.full_name = cu_full_name;
      x.kernel_name = xcux.getKernelName();
      x.name = xcux.getName();
      x.cu_mask = (1u << xcux.getCUIdx());
      x.device_id = xcux.getFPGAIdx();
      x.core_id = xcux.getCUIdx();
      x.fingerprint = my_get_fingerprint(cu_full_name, x.handle, x.cu_index,
                                         x.cu_base_addr);
      x.butler_handle = alloc.myHandle;

      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE)) << to_string(x);
      cu_idx = cu_idx + 1;
      auto r = xclOpenContext(x.handle, &uuid[0], x.cu_index, true);
      PCHECK(r == 0) << "cannot open context! " << to_string(x);
    }
  }
  if (handles_.empty()) {
    LOG(WARNING) << "cannot obtain any device, maybe used by another process.";
  }
}

XrtDeviceHandleImp::~XrtDeviceHandleImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "handle is destroyed " << (void*)this;
  std::set<butler::handle> handles;
  auto uuid = binstream_->get_uuid();
  for (const auto& handle : handles_) {
    auto& x = handle.second;
    auto r = xclCloseContext(x.handle, &uuid[0], x.cu_index);
    PCHECK(r == 0) << "cannot close context! "
                   << " cu_mask " << x.cu_mask    //
                   << " cu_index " << x.cu_index  //
                   << " xcu_addr " << std::hex << "0x" << x.cu_base_addr
                   << std::dec  //
        ;
    xclClose(x.handle);
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << " cu_handle " << x.handle                                      //
        << " cu_mask " << x.cu_mask                                       //
        << " cu_index " << x.cu_index                                     //
        << " cu_addr " << std::hex << "0x" << x.cu_base_addr << std::dec  //
        ;
    if (handles.find(x.butler_handle) == handles.end()) {
      auto releaseResult = client_->releaseResources(x.butler_handle);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "releaseResult= " << releaseResult;
      handles.insert(x.butler_handle);
    }
  }
}

const DeviceObject& XrtDeviceHandleImp::find_cu(const std::string& cu_name,
                                                size_t core_idx) const {
  return const_cast<XrtDeviceHandleImp*>(this)->find_cu(cu_name, core_idx);
}
DeviceObject& XrtDeviceHandleImp::find_cu(const std::string& cu_name,
                                          size_t core_idx) {
  auto cnt = 0u;
  DeviceObject* ret = nullptr;
  for (auto& x : handles_) {
    if (start_with(x.first, cu_name)) {
      if (cnt == core_idx) {
        ret = &x.second;
        break;
      }
      cnt = cnt + 1;
    }
  }
  UNI_LOG_CHECK(ret != nullptr, VART_XRT_NULL_PTR)
                        << "cannot found cu handle!"
                        << "cu_name " << cu_name << " "    //
                        << "core_idx " << core_idx << " "  //
      ;
  return *ret;
}

xclDeviceHandle XrtDeviceHandleImp::get_handle(const std::string& cu_name,
                                               size_t core_idx) {
  return find_cu(cu_name, core_idx).handle;
};

unsigned int XrtDeviceHandleImp::get_cu_mask(const std::string& cu_name,
                                             size_t core_idx) const {
  return find_cu(cu_name, core_idx).cu_mask;
}

uint64_t XrtDeviceHandleImp::get_cu_addr(const std::string& cu_name,
                                         size_t core_idx) const {
  return find_cu(cu_name, core_idx).cu_base_addr;
}

unsigned int XrtDeviceHandleImp::get_num_of_cus(
    const std::string& cu_name) const {
  auto cnt = 0u;
  for (auto& x : handles_) {
    if (start_with(x.first, cu_name)) {
      cnt = cnt + 1;
    }
  }
  return cnt;
}
std::string XrtDeviceHandleImp::get_cu_full_name(const std::string& cu_name,
                                                 size_t device_core_idx) const {
  return find_cu(cu_name, device_core_idx).full_name;
}
std::string XrtDeviceHandleImp::get_cu_kernel_name(
    const std::string& cu_name, size_t device_core_idx) const {
  return find_cu(cu_name, device_core_idx).kernel_name;
}
std::string XrtDeviceHandleImp::get_instance_name(
    const std::string& cu_name, size_t device_core_idx) const {
  return find_cu(cu_name, device_core_idx).name;
}
size_t XrtDeviceHandleImp::get_device_id(const std::string& cu_name,
                                         size_t device_core_idx) const {
  return find_cu(cu_name, device_core_idx).device_id;
}

size_t XrtDeviceHandleImp::get_core_id(const std::string& cu_name,
                                       size_t device_core_idx) const {
  return find_cu(cu_name, device_core_idx).core_id;
}

uint64_t XrtDeviceHandleImp::get_fingerprint(const std::string& cu_name,
                                             size_t device_core_idx) const {
  return find_cu(cu_name, device_core_idx).fingerprint;
}

}  // namespace

// when we need one xrt-device-handle per xrt-cu, not sharing, we need the
// following function
/* std::unique_ptr<xir::XrtDeviceHandle> my_xir_device_handle_create() {
  return std::make_unique<XrtDeviceHandleImp>();
  } */

namespace {
static struct Registar {
  Registar() {
    if (ENV_PARAM(XLNX_ENABLE_XIP)) {
      xir::XrtDeviceHandle::registar("02_xip_butler", []() {
        return vitis::ai::WeakSingleton<XrtDeviceHandleImp>::create();
      });
    }
  }
} g_registar;
}  // namespace

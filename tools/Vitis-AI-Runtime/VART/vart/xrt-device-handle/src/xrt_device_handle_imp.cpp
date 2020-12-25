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
#include "./xrt_device_handle_imp.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <utility>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/simple_config.hpp"
#include "vitis/ai/weak.hpp"
#include "xrt_xcl_read.hpp"
DEF_ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE, "0");
DEF_ENV_PARAM(XLNX_DISABLE_CHECK_DEVICE_TYPE, "1");
DEF_ENV_PARAM(XLNX_DISABLE_LOAD_XCLBIN, "0");
DEF_ENV_PARAM_2(XLNX_ENABLE_DEVICES, "ALL", std::string);
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string);
DEF_ENV_PARAM_2(XLNX_DDR_OR_HBM, "", std::vector<std::string>);

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

static bool start_with(const std::string& a, const std::string& b) {
  return a.find(b) == 0u;
}

static const std::string my_get_kernel_name(const std::string& name) {
  auto pos = name.find_first_of(':');
  auto ret = name;
  if (pos != std::string::npos) {
    ret = name.substr(0, pos);
  }
  return ret;
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
  auto cu_offset = 0x1F0;
  auto read_result = xrtXclRead(handle, ip_index, cu_offset, base, &l_value);
  CHECK_EQ(read_result, 0) << "xclRead has error!";
  read_result =
      xrtXclRead(handle, ip_index, cu_offset + sizeof(l_value), base, &h_value);
  CHECK_EQ(read_result, 0) << "xclRead has error!";
  ret = h_value;
  ret = (ret << 32) + l_value;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << full_cu_name << " fingerprint: 0x" << std::hex << ret << std::dec
      << std::hex << " 0x" << h_value << " 0x" << l_value << std::dec;
  return ret;
}

static std::vector<xclDeviceInfo2> get_all_device_info(size_t num_of_devices) {
  std::vector<xclDeviceInfo2> devices;
  devices.reserve(num_of_devices);
  for (auto deviceIndex = 0u; deviceIndex < num_of_devices; ++deviceIndex) {
    auto handle = xclOpen(deviceIndex, NULL, XCL_INFO);
    xclDeviceInfo2 deviceInfo;
    CHECK_EQ(xclGetDeviceInfo2(handle, &deviceInfo), 0)
        << "Unable to obtain device information";
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE) >= 2)
        << "DSA = " << deviceInfo.mName << "\n"                         //
        << "PCIe = GEN" << deviceInfo.mPCIeLinkSpeed << "x"             //
        << deviceInfo.mPCIeLinkWidth << "\n"                            //
        << "OCL Frequency = " << deviceInfo.mOCLFrequency[0] << " MHz"  //
        << "\n"                                                         //
        << "DDR Bank = " << deviceInfo.mDDRBankCount << "\n"            //
        << "Device Temp = " << deviceInfo.mOnChipTemp << " C\n"         //
        << "DeviceVersion = " << std::hex << "0x" << deviceInfo.mDeviceVersion
        << std::dec << "\n"                                                  //
        << "MIG Calibration = " << std::boolalpha << deviceInfo.mMigCalib    //
        << "\n"                                                              //
        << "mMagic = " << deviceInfo.mMagic << "\n"                          //
        << "mHALMajorVersion = " << deviceInfo.mHALMajorVersion << "\n"      //
        << "mHALMinorVersion = " << deviceInfo.mHALMinorVersion << "\n"      //
        << "mVendorId = " << deviceInfo.mVendorId << "\n"                    //
        << "mDeviceId = " << deviceInfo.mDeviceId << "\n"                    //
        << "mSubsystemId = " << deviceInfo.mSubsystemId << "\n"              //
        << "mSubsystemVendorId = " << deviceInfo.mSubsystemVendorId << "\n"  //
        << "mDeviceVersion = " << deviceInfo.mDeviceVersion << "\n"          //
        << "DDR memory size = " << deviceInfo.mDDRSize << "\n"               //
        << "Minimum data alignment requirement for host buffers = "
        << deviceInfo.mDataAlignment << "\n"  //
        << "Total unused/available DDR memory =  " << deviceInfo.mDDRFreeSize
        << " \n"  //
        << "Minimum DMA buffer size = " << deviceInfo.mMinTransferSize
        << "\n "  //
        ;
    devices.push_back(deviceInfo);
    xclClose(handle);
  }
  return devices;
}

static std::pair<std::string, std::string> split_at(const std::string& str,
                                                    const char delimiter) {
  auto ret = std::pair<std::string, std::string>();

  std::istringstream names_istr(str);
  std::getline(names_istr, std::get<0>(ret), delimiter);
  std::getline(names_istr, std::get<1>(ret), delimiter);
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

static std::string do_detect_ddr_or_hbm(const std::string& dsa_name) {
  std::string name = dsa_name;
  std::transform(name.begin(), name.end(), name.begin(),
                 [](std::string::value_type c) -> std::string::value_type {
                   return (std::string::value_type)std::toupper(c);
                 });
  if (name.find("_U50_") != name.npos) {
    return "HBM";
  } else if (name.find("_U50LV_") != name.npos) {
    return "HBM";
  } else if (name.find("_U250_") != name.npos) {
    return "HBM";
  } else if (name.find("_U280_") != name.npos) {
    return "HBM";
  }
  return "DDR";
}

XrtDeviceHandleImp::XrtDeviceHandleImp() {
  static_assert(sizeof(xuid_t) == SIZE_OF_UUID, "ERROR: UUID size mismatch");
  auto num_of_devices = xclProbe();
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "probe num of devices = " << num_of_devices;
  auto devices = get_all_device_info(num_of_devices);
  CHECK_EQ(devices.size(), num_of_devices);
  auto filename = get_dpu_xclbin();
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "open firmware " << filename;
  binstream_ = std::make_unique<xir::XrtBinStream>(filename);
  auto dsa_name = binstream_->get_dsa();

  auto device_id_list = get_device_id_list(num_of_devices);
  auto& xlnx_ddr_or_hbm = ENV_PARAM(XLNX_DDR_OR_HBM);
  auto detect_ddr_or_hbm = xlnx_ddr_or_hbm.empty();
  if (detect_ddr_or_hbm) {
    xlnx_ddr_or_hbm.resize(num_of_devices);
  }
  for (const auto& deviceIndex : device_id_list) {
    if (!ENV_PARAM(XLNX_DISABLE_CHECK_DEVICE_TYPE)) {
#ifdef ENABLE_CLOUD
      if (std::strcmp(devices[deviceIndex].mName, dsa_name.c_str()) != 0) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
            << "deviceIndex " << deviceIndex << " "  //
            << "devices[deviceIndex].mName " << devices[deviceIndex].mName
            << " "                             //
            << "dsa_name " << dsa_name << " "  //
            ;
        continue;
      }
#endif
    }
    if (detect_ddr_or_hbm) {
      xlnx_ddr_or_hbm[deviceIndex] =
          do_detect_ddr_or_hbm(devices[deviceIndex].mName);
    }
    auto lock_file_name =
        std::string("/tmp/vart_device_") + std::to_string(deviceIndex);
    auto lock_pid_file_name =
        std::string("/tmp/vart_device_") + std::to_string(deviceIndex) + ".pid";
    auto mtx = std::make_unique<vitis::ai::FileLock>(lock_file_name);
    mtx_.push_back(std::move(mtx));
    auto lock = std::make_unique<std::unique_lock<vitis::ai::FileLock>>(
        *(mtx_.back().get()), std::try_to_lock_t());
    if (!lock->owns_lock()) {
      int pid = 0;
      pid = ((std::ifstream(lock_pid_file_name) >> pid).good()) ? pid : -1;
      LOG(INFO) << "waiting for process [" << pid
                << "] to release the resource:" << lock_file_name;
      lock->lock();
    }
    file_lock_.push_back(std::move(lock));
    if (!(std::ofstream(lock_pid_file_name) << getpid() << std::endl).good()) {
      LOG(INFO) << "can not write process id to " << lock_file_name;
    }
    auto fd =
        open(lock_pid_file_name.c_str(), FD_CLOEXEC | O_WRONLY | O_CREAT, 0777);
    fchmod(fd, 0666);
    close(fd);

    auto handle = xclOpen(deviceIndex, NULL, XCL_INFO);
    if (!ENV_PARAM(XLNX_DISABLE_LOAD_XCLBIN)) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "load xclbin begin, device " << deviceIndex;
      binstream_->burn(handle);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "load xclbin success, device " << deviceIndex;
    } else {
      LOG(INFO) << "no load xclbin";
    }
    auto uuid = binstream_->get_uuid();
    xclClose(handle);
    for (auto i = 0u; i < binstream_->get_num_of_cu(); ++i) {
      auto cu_full_name = binstream_->get_cu(i);
      auto kernel_name = my_get_kernel_name(cu_full_name);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "cu[" << i << "] = " << cu_full_name << " cu_name=" << kernel_name;
      auto& x = handles_[cu_full_name + ":" + std::to_string(deviceIndex)];
      x.cu_base_addr = binstream_->get_cu_base_addr(i);
      x.handle = xclOpen(deviceIndex, NULL, XCL_INFO);
      int cu_index = -1;
#ifdef HAS_xclIPName2Index_3
      uint32_t dummy_id;
      cu_index = xclIPName2Index(x.handle, cu_full_name.c_str(), &dummy_id);
#endif
#ifdef HAS_xclIPName2Index_2
      cu_index = xclIPName2Index(x.handle, cu_full_name.c_str());
#endif
      CHECK_NE(cu_index, -1) << "cannot get cu_index. cu name=" << cu_full_name;
      x.cu_index = (size_t)cu_index;
      x.ip_index = i;
      x.full_name = cu_full_name;
      std::tie(x.kernel_name, x.instance_name) = split_at(cu_full_name, ':');
      // binstream_->burn(x.handle);  // need to load bin to get reg map.
      x.cu_mask = (1u << x.cu_index);
      x.device_id = deviceIndex;
      x.core_id = i;
      x.fingerprint = my_get_fingerprint(cu_full_name, x.handle, x.cu_index,
                                         x.cu_base_addr);
      x.uuid = uuid;
      auto r = xclOpenContext(x.handle, &uuid[0], x.cu_index, true);
      PCHECK(r == 0) << "cannot open context! "
                     << "cu_index " << x.cu_index << " "          //
                     << "cu_base_addr " << x.cu_base_addr << " "  //
                     << "fingerprint " << std::hex << "0x" <<     //
          x.fingerprint << std::dec << " "                        //
          ;
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << " cu_handle " << x.handle                                      //
          << " device_id " << x.device_id << " "                            //
          << " full_name " << x.full_name << " "                            //
          << " kernel_name " << x.kernel_name << " "                        //
          << " instance_name " << x.instance_name << " "                    //
          << " cu_mask " << x.cu_mask                                       //
          << " cu_index " << x.cu_mask                                      //
          << " cu_addr " << std::hex << "0x" << x.cu_base_addr << std::dec  //
          << " fingerprint " << std::hex << "0x" <<                         //
          x.fingerprint << std::dec << " "                                  //
          ;
    }
  }
  // TODO : check handles_ is not null
  PCHECK(handles_.size() > 0) << "No device can use !";
}  // namespace

XrtDeviceHandleImp::~XrtDeviceHandleImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "handle is destroyed " << (void*)this << "num of devices "
      << handles_.size();
  if (handles_.empty()) {
    return;
  }
  auto uuid = binstream_->get_uuid();
  std::set<std::string> deleted{};
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
    auto deviceIndex = x.device_id;
    auto lock_pid_file_name =
        std::string("/tmp/vart_device_") + std::to_string(deviceIndex) + ".pid";
    if (deleted.find(lock_pid_file_name) == deleted.end()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "delete lock file " << lock_pid_file_name;
      unlink(lock_pid_file_name.c_str());
      deleted.insert(lock_pid_file_name);
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
  CHECK(ret != nullptr) << "cannot found cu handle!"
                        << "cu_name " << cu_name << " "    //
                        << "core_idx " << core_idx << " "  //
      ;
  return *ret;
}

xclDeviceHandle XrtDeviceHandleImp::get_handle(const std::string& cu_name,
                                               size_t core_idx) {
  return find_cu(cu_name, core_idx).handle;
};

size_t XrtDeviceHandleImp::get_cu_index(const std::string& cu_name,
                                        size_t core_idx) const {
  return find_cu(cu_name, core_idx).cu_index;
}

size_t XrtDeviceHandleImp::get_ip_index(const std::string& cu_name,
                                        size_t core_idx) const {
  return find_cu(cu_name, core_idx).ip_index;
}

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
  return find_cu(cu_name, device_core_idx).instance_name;
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
unsigned int XrtDeviceHandleImp::get_bank_flags(const std::string& cu_name,
                                                size_t device_core_idx) const {
  // TODO return available banks. and DEV_ONLY
  //#if CROSSCOMPILING
  return XCL_BO_FLAGS_CACHEABLE;
  //#else
  //  return XCL_BO_FLAGS_DEV_ONLY;
  //#endif
}

std::array<unsigned char, SIZE_OF_UUID> XrtDeviceHandleImp::get_uuid(
    const std::string& cu_name, size_t device_core_idx) const {
  return find_cu(cu_name, device_core_idx).uuid;
}
}  // namespace

// when we need one xrt-device-handle per xrt-cu, not sharing, we need the
// following function
/* std::unique_ptr<xir::XrtDeviceHandle> my_xir_device_handle_create() {
  return std::make_unique<XrtDeviceHandleImp>();
  }*/

namespace {
static struct Registar {
  Registar() {
    xir::XrtDeviceHandle::registar("03_xrt_edge", []() {
      return vitis::ai::WeakSingleton<XrtDeviceHandleImp>::create();
    });
  }
} g_registar;
}  // namespace

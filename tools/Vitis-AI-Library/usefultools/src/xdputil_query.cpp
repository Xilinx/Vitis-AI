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

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/target_factory.hpp>
#include <xir/device_memory.hpp>
#include <xir/xrt_device_handle.hpp>

#include "tools_extra_ops.hpp"

template <typename T>
uint64_t data_slice(T source, int begin, int end) {
  return (source >> begin) & ((T(1) << int(end - begin)) - 1);
}

template <typename T>
uint64_t data_slice(T source_h, T source_l) {
  return (uint64_t(source_h) << 32) + uint64_t(source_l);
}

class base_device;
struct device_info_struct {
  uint64_t fingerprint;
  std::unique_ptr<base_device> device;
  std::string dpu_arch_type;
  std::string dpu_arch;
  std::string cu_name;
  void* cu_handle;
  uint64_t cu_addr;
  uint64_t cu_idx;
  uint64_t core_idx;
  uint64_t device_id;
  uint64_t cu_mask;
};

class base_device {
 public:
  virtual py::dict get_public_data(const device_info_struct& info,
                                   size_t core_count) {
    return py::dict();
  }

  virtual py::dict get_private_data(const device_info_struct& info);
  virtual py::dict read_dpu_register(const device_info_struct& info) {
    py::dict res;
    res["Unsupported IP, fingerprint"] = to_string(info.fingerprint, hex);
    res["name"] = "DPU Registers Core " + to_string(info.core_idx);
    return res;
  }
};

class DPUCZDX8G_VIVADO_FLOW_DEVICE : public base_device {
 public:
  DPUCZDX8G_VIVADO_FLOW_DEVICE(xir::DeviceMemory* d) : device_memory(d) {}
  xir::DeviceMemory* get_device_memory() { return device_memory; }
  virtual py::dict get_private_data(const device_info_struct& info);
  virtual py::dict read_dpu_register(const device_info_struct& info);

 private:
  xir::DeviceMemory* device_memory;
};

class DPUCZDX8G_device : public base_device {
 public:
  virtual py::dict get_public_data(const device_info_struct& info,
                                   size_t core_count);
  virtual py::dict read_dpu_register(const device_info_struct& info);
  virtual py::dict get_private_data(const device_info_struct& info);
  DPUCZDX8G_device(std::vector<std::string> key_i = {},
                   std::vector<uint32_t> addr_i = {});

 private:
  std::vector<std::string> key;
  std::vector<uint32_t> addr;
};

class DPUCVDX8G_device : public DPUCZDX8G_device {
 public:
  virtual py::dict get_private_data(const device_info_struct& info);
  DPUCVDX8G_device();

 private:
  std::vector<std::string> create_key();
  std::vector<uint32_t> create_addr();
};

class DPUCAHX8H_device : public base_device {
 public:
  virtual py::dict get_public_data(const device_info_struct& info,
                                   size_t core_count);
  virtual py::dict read_dpu_register(const device_info_struct& info);
  DPUCAHX8H_device();

 private:
  std::vector<std::string> key;
  std::vector<uint32_t> addr;
};

void device_info(xir::XrtDeviceHandle* h,
                 std::vector<device_info_struct>& infos) {
  __TIC__(DEVICE_INFO)

  auto cu_name = std::string("");
  infos.resize(h->get_num_of_cus(cu_name));
  std::vector<py::dict> result;
  int i = 0;
  for (auto& info : infos) {
    info.device_id = h->get_device_id(cu_name, i);
    info.cu_name = h->get_cu_full_name(cu_name, i);
    info.fingerprint = h->get_fingerprint(cu_name, i);
    info.cu_handle = h->get_handle(cu_name, i);
    info.cu_addr = h->get_cu_addr(cu_name, i);
    info.cu_idx = h->get_cu_index(cu_name, i);
    info.core_idx = i;
    info.cu_mask = h->get_cu_mask(cu_name, i);
    if (info.fingerprint) {
      auto target = vitis::ai::target_factory()->create(info.fingerprint);
      info.dpu_arch_type = target.type();
      info.dpu_arch = target.name();
    }
    if (info.dpu_arch_type == "DPUCZDX8G") {
      info.device = std::unique_ptr<base_device>(new DPUCZDX8G_device());
    } else if (info.dpu_arch_type == "DPUCVDX8G") {
      info.device = std::unique_ptr<base_device>(new DPUCVDX8G_device());
    } else if (info.dpu_arch_type == "DPUCAHX8H") {
      info.device = std::unique_ptr<base_device>(new DPUCAHX8H_device());
    } else {
      info.device = std::unique_ptr<base_device>(new base_device());
      LOG(ERROR) << "Unsupported platform fingerprint: " << info.fingerprint
                 << ", cu_idx: " << info.core_idx;
    }
    i++;
  }
  __TOC__(DEVICE_INFO)
}

void get_vivado_infos(xir::DeviceMemory* d,
                      std::vector<device_info_struct>& infos) {
  uint64_t fingerprint;
  uint64_t fingerprint_h;
  d->download(&fingerprint, 0x8F0001F0, 4);
  d->download(&fingerprint_h, 0x8F0001F4, 4);
  fingerprint = (fingerprint_h << 32) + fingerprint;
  infos.resize(1);
  infos[0].fingerprint = fingerprint;

  infos[0].device = std::make_unique<DPUCZDX8G_VIVADO_FLOW_DEVICE>(d);
  infos[0].cu_idx = 0;
  infos[0].cu_addr = 0x8F000000;
  if (infos[0].fingerprint) {
    auto target = vitis::ai::target_factory()->create(infos[0].fingerprint);
    infos[0].dpu_arch = target.name();
  }
}

py::dict create_vai_version() {
  std::vector<std::string> so_list = {"libxir.so", "libvart-runner.so",
                                      "libvitis_ai_library-dpu_task.so"};
  auto so_res = xilinx_version(so_list);
  py::dict res;
  for (auto i = 0u; i < so_list.size(); i++) {
    res[so_list[i].c_str()] = so_res[i];
  }
  res["target_factory"] =
      std::string(vitis::ai::TargetFactory::get_lib_name()) + " " +
      std::string(vitis::ai::TargetFactory::get_lib_id());

  return res;
}

py::dict xdputil_query() {
  // 1.device_info
  std::shared_ptr<xir::XrtDeviceHandle> h;
  std::unique_ptr<xir::DeviceMemory> device_memory;
  std::vector<device_info_struct> infos;
  if (!access("/dev/dpu", F_OK)) {
    // vivado flow
    device_memory = xir::DeviceMemory::create((size_t)0ull);
    get_vivado_infos(device_memory.get(), infos);
  } else {
    h = xir::XrtDeviceHandle::get_instance();
    device_info(h.get(), infos);
  }
  py::dict res;
  // 2. DPU IP Spec
  for (auto& info : infos) {
    if (info.device) {
      res["DPU IP Spec"] = info.device->get_public_data(info, infos.size());
      break;
    }
  }
  // 3. VAI Version
  res["VAI Version"] = create_vai_version();
  // res = whoami();
  // 4. kernels
  std::vector<py::dict> kernels;
  for (auto& info : infos) {
    if (info.device) {
      kernels.push_back(info.device->get_private_data(info));
    }
  }
  res["kernels"] = kernels;
  return res;
}

py::dict xdputil_status() {
  // 1.device_info
  std::shared_ptr<xir::XrtDeviceHandle> h;
  std::unique_ptr<xir::DeviceMemory> device_memory;
  std::vector<device_info_struct> infos;
  if (!access("/dev/dpu", F_OK)) {
    // vivado flow
    device_memory = xir::DeviceMemory::create((size_t)0ull);
    get_vivado_infos(device_memory.get(), infos);
  } else {
    h = xir::XrtDeviceHandle::get_instance();
    device_info(h.get(), infos);
  }
  // 2. DPU registers
  std::vector<py::dict> registers;
  for (auto& info : infos) {
    if (info.device) {
      registers.push_back(info.device->read_dpu_register(info));
    }
  }
  py::dict res;
  res["kernels"] = registers;
  return res;
}

// base_device
py::dict base_device::get_private_data(const device_info_struct& info) {
  py::dict res;
  res["fingerprint"] = to_string(info.fingerprint, hex);
  res["cu_handle"] = to_string((uint64_t)info.cu_handle, hex);
  res["cu_idx"] = info.core_idx;
  res["cu_mask"] = info.cu_mask;
  res["cu_name"] = info.cu_name;
  res["device_id"] = info.device_id;
  res["cu_addr"] = to_string(info.cu_addr, hex);
  res["name"] = "DPU Core " + to_string(info.core_idx);
  res["DPU Arch"] = info.dpu_arch;
  return res;
}

// DPUCZDX8G_VIVADO_FLOW_DEVICE
py::dict DPUCZDX8G_VIVADO_FLOW_DEVICE::get_private_data(
    const device_info_struct& info) {
  py::dict res;
  res["fingerprint"] = to_string(info.fingerprint, hex);
  res["cu_idx"] = info.cu_idx;
  res["DPU Arch"] = info.dpu_arch;
  res["cu_addr"] = to_string(info.cu_addr, hex);
  res["is_vivado_flow"] = true;

  res["name"] = "DPU Core " + to_string(info.cu_idx);
  uint32_t frequency;
  device_memory->download(&frequency, info.cu_addr + 0xF00004, 4);
  res["DPU Frequency (MHz)"] = data_slice(frequency, 1, 11);
  return res;
}

py::dict DPUCZDX8G_VIVADO_FLOW_DEVICE::read_dpu_register(
    const device_info_struct& info) {
  py ::dict res;
  // 1. common_registers
  py::dict common_registers;

  uint32_t tmp;
  auto read_reg = [&](uint32_t addr) {
    device_memory->download(&tmp, info.cu_addr + addr, 4);
    return tmp;
  };
  common_registers["LOAD START"] = read_reg(0x280);
  common_registers["LOAD END"] = read_reg(0x270);
  common_registers["SAVE START"] = read_reg(0x27C);
  common_registers["SAVE END"] = read_reg(0x26C);
  common_registers["CONV START"] = read_reg(0x278);
  common_registers["CONV END"] = read_reg(0x268);
  common_registers["MISC START"] = read_reg(0x274);
  common_registers["MISC END"] = read_reg(0x264);
  common_registers["PROF_VALUE"] = read_reg(0x214);
  common_registers["PROF_NUM"] = read_reg(0x218);

  // HP
  tmp = read_reg(0x200);
  common_registers["HP_AWCOUNT_MAX"] = data_slice(tmp, 24, 32);
  common_registers["HP_ARCOUNT_MAX"] = data_slice(tmp, 16, 24);
  common_registers["HP_AWLEN"] = data_slice(tmp, 8, 16);
  common_registers["HP_ARLEN"] = data_slice(tmp, 0, 8);

  // DPU code addr
  device_memory->download(&tmp, info.cu_addr + 0x20C, 4);
  common_registers["ADDR_CODE"] = to_string(tmp, hex);
  res["common_registers"] = common_registers;

  // 2. addrs_registers
  py::dict addrs_registers;
  for (auto j = 0u; j < 8; j++) {
    addrs_registers[(string("dpu0_base_addr_") + to_string(j)).c_str()] =
        to_string(data_slice(read_reg(0x228 + j * 8), read_reg(0x224 + j * 8)),
                  hex);
  }
  res["addrs_registers"] = addrs_registers;

  // 3. name
  res["name"] = "DPU Registers Core " + to_string(info.cu_idx);
  return res;
}

// DPUCVDX8G_device
py::dict DPUCVDX8G_device::get_private_data(const device_info_struct& info) {
  auto res = DPUCZDX8G_device::get_private_data(info);

  auto read_res =
      read_register(info.cu_handle, info.cu_idx, info.cu_addr, {0x134});
  //                                                           "BATCH_N"
  res["DPU Batch Number"] = data_slice(read_res[0], 0, 4);
  return res;
}

std::vector<std::string> DPUCVDX8G_device::create_key() {
  std::vector<std::string> key;
  for (auto i = 0u; i < 6; i++) {
    for (auto j = 0u; j < 4; j++) {
      key.push_back("dpu" + to_string(i) + "_base_addr_" + to_string(j));
      key.push_back("dpu" + to_string(i) + "_base_addr_" + to_string(j) + "_h");
    }
  }
  return key;
}

std::vector<uint32_t> DPUCVDX8G_device::create_addr() {
  std::vector<uint32_t> addr;
  for (auto i = 0u; i < 6; i++) {
    for (auto j = 0u; j < 4; j++) {
      addr.push_back(0x200 + (i * 4 + j) * 8);
      addr.push_back(0x204 + (i * 4 + j) * 8);
    }
  }
  return addr;
}

DPUCVDX8G_device::DPUCVDX8G_device()
    : DPUCZDX8G_device(create_key(), create_addr()) {}

// DPUCZDX8G_device
py::dict DPUCZDX8G_device::get_public_data(const device_info_struct& info,
                                           size_t core_count) {
  auto read_res = read_register(info.cu_handle, info.cu_idx, info.cu_addr,
                                {0x20, 0x108, 0x24, 0x100, 0x104});
  //"SYS","SUB_VERSION","TIMESTAMP","GIT_COMMIT_ID","GIT_COMMIT_TIME"
  py::dict dpu_inf;
  // GIT_COMMIT_ID
  dpu_inf["git commit id"] = to_string(data_slice(read_res[3], 0, 28), hex, "");
  // GIT_COMMIT_TIME
  dpu_inf["git commit time"] = data_slice(read_res[4], 0, 32);

  // TIMESTAMP yyyy-MM-dd HH-mm-ss
  std::string tmp, timestamp;
  timestamp = "20" + to_string(data_slice(read_res[2], 24, 32)) + "-";
  tmp = to_string(data_slice(read_res[2], 20, 24));
  timestamp += ((tmp.size() < 2) ? "0" + tmp : tmp) + "-";
  tmp = to_string(data_slice(read_res[2], 12, 20));
  timestamp += ((tmp.size() < 2) ? "0" + tmp : tmp) + " ";
  tmp = to_string(data_slice(read_res[2], 4, 12));
  timestamp += ((tmp.size() < 2) ? "0" + tmp : tmp) + "-";
  tmp = to_string(data_slice(read_res[2], 0, 4) * 15);
  timestamp += ((tmp.size() < 2) ? "0" + tmp : tmp) + "-00";
  dpu_inf["generation timestamp"] = timestamp;

  // SYS
  tmp = to_string(data_slice(read_res[0], 24, 32), hex, "");
  tmp = (tmp.size() < 2) ? "0" + tmp : tmp;
  dpu_inf["IP version"] = string("v") + tmp[0] + "." + tmp[1] + "." +
                          to_string(data_slice(read_res[1], 12, 20), hex, "");
  tmp = to_string(data_slice(read_res[1], 0, 12), hex, "");
  tmp = (tmp.size() < 3) ? "0" + tmp : tmp;
  tmp = (tmp.size() < 3) ? "0" + tmp : tmp;
  dpu_inf["DPU Target Version"] =
      string("v") + tmp[0] + "." + tmp[1] + "." + tmp[2];

  vector<string> regmap_version = {"Initial version", "1toN version",
                                   "1to1 version"};
  dpu_inf["regmap"] = (data_slice(read_res[0], 0, 8) < 3)
                          ? regmap_version[data_slice(read_res[0], 0, 8)]
                          : "null";
  dpu_inf["DPU Core Count"] = core_count;
  return dpu_inf;
}

py::dict DPUCZDX8G_device::get_private_data(const device_info_struct& info) {
  auto res = base_device::get_private_data(info);

  auto read_res = read_register(info.cu_handle, info.cu_idx, info.cu_addr,
                                {0x20, 0x28, 0x118, 0x120});
  //                            "SYS","FREQ","LOAD","SAVE"
  vector<string> ip_type = {"",       "DPU",  "softmax", "sigmoid",
                            "resize", "SMFC", "YRR"};
  res["IP Type"] = ip_type[data_slice(read_res[0], 16, 24)];
  res["DPU Frequency (MHz)"] = data_slice(read_res[1], 0, 12);
  res["XRT Frequency (MHz)"] = data_slice(read_res[1], 12, 24);
  res["Save Parallel"] = data_slice(read_res[3], 0, 4);
  res["Load Parallel"] = data_slice(read_res[2], 0, 4);
  res["Load minus mean"] = data_slice(read_res[2], 4, 8) ? "enable" : "disable";
  res["Load augmentation"] =
      data_slice(read_res[2], 8, 12) ? "enable" : "disable";
  return res;
}

DPUCZDX8G_device::DPUCZDX8G_device(std::vector<std::string> key_i,
                                   std::vector<uint32_t> addr_i) {
  key = {"AP_REG",  //
         "LOAD START", "LOAD END",     "SAVE START",  "SAVE END",
         "CONV START", "CONV END",     "MISC START",  "MISC END",  //
         "HP_BUS",     "INSTR_ADDR_L", "INSTR_ADDR_H"};
  addr = {0x00,                        //
          0x180, 0x184, 0x190, 0x194,  //
          0x188, 0x18C, 0x198, 0x19C,  //
          0x48,  0x50,  0x54};
  if (key_i.empty()) {
    for (auto j = 0u; j < 8; j++) {
      key.push_back("dpu0_base_addr_" + to_string(j));
      addr.push_back(0x60 + j * 8);
      key.push_back("dpu0_base_addr_" + to_string(j) + "_h");
      addr.push_back(0x64 + j * 8);
    }
  } else {
    key.insert(key.end(), key_i.begin(), key_i.end());
    addr.insert(addr.end(), addr_i.begin(), addr_i.end());
  }
}

py::dict DPUCZDX8G_device::read_dpu_register(const device_info_struct& info) {
  py ::dict res;
  // 1. common_registers
  py::dict common_registers;
  auto registers =
      read_register(info.cu_handle, info.cu_idx, info.cu_addr, addr);
  auto idx = 0u;
  // AP_REG
  std::map<uint32_t, std::string> ap_status = {
      {0b0001, "start"}, {0b0010, "done"}, {0b0100, "idle"}, {0b1000, "ready"}};
  std::map<uint32_t, std::string> ap_reset_status = {
      {0b00, ""}, {0b01, "soft reset start"}, {0b10, "soft reset done"}};

  common_registers["AP status"] =
      ap_status[data_slice(registers[idx], 0, 4)] +
      ap_reset_status[data_slice(registers[idx], 5, 7)];
  idx++;
  for (auto i = idx; i < idx + 8; i++) {
    common_registers[key[i].c_str()] = registers[i];
  }
  idx += 8;
  // HP
  common_registers["HP_AWCOUNT_MAX"] = data_slice(registers[idx], 24, 32);
  common_registers["HP_ARCOUNT_MAX"] = data_slice(registers[idx], 16, 24);
  common_registers["HP_AWLEN"] = data_slice(registers[idx], 8, 16);
  common_registers["HP_ARLEN"] = data_slice(registers[idx++], 0, 8);
  common_registers["ADDR_CODE"] =
      to_string(data_slice(registers[idx + 1], registers[idx]), hex);
  idx += 2;
  res["common_registers"] = common_registers;
  // 2. addrs_registers
  py::dict addrs_registers;

  for (; idx + 1 < key.size(); idx += 2) {
    addrs_registers[key[idx].c_str()] =
        to_string(data_slice(registers[idx + 1], registers[idx]), hex);
  }
  res["addrs_registers"] = addrs_registers;
  // 3. name
  res["name"] = "DPU Registers Core " + to_string(info.core_idx);
  return res;
}

// DPUCAHX8H_device

py::dict DPUCAHX8H_device::get_public_data(const device_info_struct& info,
                                           size_t core_count) {
  auto read_res =
      read_register(info.cu_handle, info.cu_idx, info.cu_addr, {0x1F0, 0x1F4});

  auto hard_ver = data_slice(read_res[1], read_res[0]);
  py::dict dpu_inf;
  dpu_inf["dwc"] = data_slice(hard_ver, 0, 1) ? "enable" : "disable";
  dpu_inf["leakyrelu"] = data_slice(hard_ver, 1, 2) ? "enable" : "disable";
  dpu_inf["misc_parallesim"] = data_slice(hard_ver, 2, 3) ? "2p" : "1p";
  switch (data_slice(hard_ver, 3, 5)) {
    case 0b11:
      dpu_inf["Bank Group Volume"] = "VB";
      break;
    case 0b01:
      dpu_inf["Bank Group Volume"] = "2MB BKG";
      break;
    case 0b00:
      dpu_inf["Bank Group Volume"] = "512KB BKG";
      break;
    default:
      dpu_inf["Bank Group Volume"] = "none";
      break;
  }
  dpu_inf["long weight"] =
      data_slice(hard_ver, 5, 6) ? "support" : "not support";
  dpu_inf["pooling kernel size 5x5 operation"] =
      data_slice(hard_ver, 6, 7) ? "support" : "not support";
  dpu_inf["pooling kernel size 8x8 operation"] =
      data_slice(hard_ver, 7, 8) ? "support" : "not support";
  dpu_inf["pooling kernel size 4x4 operation"] =
      data_slice(hard_ver, 8, 9) ? "support" : "not support";
  dpu_inf["pooling kernel size 6x6 operation"] =
      data_slice(hard_ver, 9, 10) ? "support" : "not support";
  dpu_inf["isa encoding"] = data_slice(hard_ver, 48, 56);
  dpu_inf["ip encoding"] = data_slice(hard_ver, 56, 64);
  dpu_inf["DPU Core Count"] = core_count;
  return dpu_inf;
}

DPUCAHX8H_device::DPUCAHX8H_device() {
  key = {"AP status",  //
         "LOAD START",     "LOAD END",     "SAVE START",  "SAVE END",
         "CONV START",     "CONV END",     "MISC START",  "MISC END",  //
         "reg_hp_setting", "INSTR_ADDR_L", "INSTR_ADDR_H"};
  addr = {0x000,                       //
          0x0A0, 0x090, 0x09C, 0x08C,  //
          0x098, 0x088, 0x094, 0x084,  //
          0x020, 0x140, 0x144};
  for (auto j = 0u; j < 8; j++) {
    key.push_back("dpu0_reg_base_addr_" + to_string(j));
    addr.push_back(0x100 + j * 8);
    key.push_back("dpu0_reg_base_addr_" + to_string(j) + "_h");
    addr.push_back(0x104 + j * 8);
  }
  for (auto i = 1u; i < 8; i++) {
    for (auto j = 0u; j < 8; j++) {
      key.push_back("dpu" + to_string(i) + "_reg_base_addr_" + to_string(j));
      addr.push_back(0x200 + (i * 4 + j) * 8);
      key.push_back("dpu" + to_string(i) + "_reg_base_addr_" + to_string(j) +
                    "_h");
      addr.push_back(0x204 + (i * 4 + j) * 8);
    }
  }
}

py::dict DPUCAHX8H_device::read_dpu_register(const device_info_struct& info) {
  py ::dict res;
  // 1. common_registers
  py::dict common_registers;
  auto registers =
      read_register(info.cu_handle, info.cu_idx, info.cu_addr, addr);
  auto idx = 0u;
  // AP_REG
  std::map<uint32_t, std::string> ap_status = {
      {0b0001, "start"}, {0b0010, "done"}, {0b0100, "idle"}};
  common_registers["AP status"] = ap_status[data_slice(registers[idx++], 0, 3)];
  for (auto i = idx; i < idx + 8; i++) {
    common_registers[key[i].c_str()] = registers[i];
  }
  idx += 8;
  common_registers["HP_COUNT_MAX"] = data_slice(registers[idx], 16, 24);
  common_registers["HP_AWLEN"] = data_slice(registers[idx], 8, 16);
  common_registers["HP_ARLEN"] = data_slice(registers[idx++], 0, 8);
  common_registers["ADDR_CODE"] =
      to_string(data_slice(registers[idx + 1], registers[idx]), hex);
  idx += 2;
  res["common_registers"] = common_registers;
  // 2. addrs_registers
  py::dict addrs_registers;

  for (auto i = 0u; i < 8; i++) {
    for (auto j = 0u; j < 8; j++) {
      addrs_registers[key[idx].c_str()] =
          to_string(data_slice(registers[idx + 1], registers[idx]), hex);
      idx += 2;
    }
  }
  res["addrs_registers"] = addrs_registers;
  // 3. name
  res["name"] = "DPU Registers Core " + to_string(info.core_idx);
  return res;
}

/*
 * Copyright 2021 Xilinx Inc.
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

#include "model_config.hpp"

#include <glog/logging.h>
#include <json-c/json.h>

#include <fstream>

#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_MODEL_CONFIG, "0");

template <int FACTOR>
static uint32_t align_to(uint32_t n) {
  auto a = n + FACTOR - 1;
  auto b = a / FACTOR;
  auto c = b * FACTOR;
  return c;
}

template <typename T>
static std::string dump_ddr_instr_layout(const std::vector<T>& layout,
                                         const std::string& filename) {
  std::ofstream of(filename);
  if (!of.is_open()) {
    LOG(WARNING) << "Couldn't open " << filename;
    return "";
  }
  for (auto val : layout) {
    of << val << "\n";
  }
  return "";
}

template <typename Map>
static std::string dump_batch_lines(const Map& batch_lines,
                                    const std::string& filename) {
  std::ofstream of(filename);
  if (!of.is_open()) {
    LOG(WARNING) << "Couldn't open " << filename;
    return "";
  }
  for (auto& [key, num_instrs] : batch_lines) {
    of << key << "\n";
    for (auto val : num_instrs) {
      of << val << "\n";
    }
  }
  return "";
}

static json_object* read_json_from_directory(
    const std::string& model_directory) {
  auto config_filename = model_directory + "/" + "config.json";
  json_object* value = json_object_from_file(config_filename.c_str());
  CHECK(value != nullptr) << "failed to read meta file! filename="
                          << config_filename;
  CHECK(json_object_is_type(value, json_type_object))
      << "not a json object. value=" << json_object_to_json_string(value);
  return value;
}

static struct LAYER_T get_layer_config(json_object* obj,
                                       const std::string& key) {
  json_object* field = nullptr;
  CHECK(json_object_object_get_ex(obj, key.c_str(), &field))
      << "no such field! key=" << key
      << ", value=" << json_object_to_json_string(obj);

  json_object* direction = nullptr;
  CHECK(json_object_object_get_ex(field, "dir", &direction))
      << "no such field! key=dir, value=" << json_object_to_json_string(field);

  json_object* size = nullptr;
  CHECK(json_object_object_get_ex(field, "size", &size))
      << "no such field! key=dir, value=" << json_object_to_json_string(field);

  struct LAYER_T l;
  l.direction = json_object_get_int(direction);
  l.size = json_object_get_int(size);

  return l;
}

static std::vector<int> get_batch_lines(json_object* obj,
                                        const std::string& key) {
  json_object* field = nullptr;
  auto ret = std::vector<int>{};
  if (!json_object_object_get_ex(obj, key.c_str(), &field)) return ret;

  auto size = json_object_array_length(field);
  ret.reserve(size);

  for (decltype(size) idx = 0; idx < size; idx++) {
    auto elt = json_object_array_get_idx(field, idx);
    CHECK(json_object_is_type(elt, json_type_int))
        << "element is not an int or array of int ! key=" << key
        << ", idx=" << idx << ", value=" << json_object_to_json_string(obj);
    ret.emplace_back(json_object_get_int(elt));
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_MODEL_CONFIG)) << ret.size();
  return ret;
}

ModelConfig::ModelConfig(const std::string& model_directory) {
  auto value = read_json_from_directory(model_directory);

  json_object* field = nullptr;
  CHECK(json_object_object_get_ex(value, "layer", &field))
      << "no such field! key=layer, value="
      << json_object_to_json_string(value);

  layers_ = json_object_object_length(field);
  for (int i = 0; i < layers_; i++) {
    json_object* ith_layer = nullptr;

    CHECK(
        json_object_object_get_ex(field, std::to_string(i).c_str(), &ith_layer))
        << "no such field! key=" << i
        << ", value=" << json_object_to_json_string(field);

    auto len = json_object_object_length(ith_layer);

    std::vector<LAYER_T> vl;
    vl.reserve(len);
    vl.emplace_back(get_layer_config(ith_layer, "load_src_reg_0"));
    vl.emplace_back(get_layer_config(ith_layer, "load_src_reg_1"));
    vl.emplace_back(get_layer_config(ith_layer, "save_dest_reg_0"));
    layer_config_.emplace_back(vl);
  }

  batch_lines_.insert(std::pair<std::string, std::vector<int>>(
      "batch4", get_batch_lines(value, "batch4_lines")));
  batch_lines_.insert(std::pair<std::string, std::vector<int>>(
      "batch3", get_batch_lines(value, "batch3_lines")));
  batch_lines_.insert(std::pair<std::string, std::vector<int>>(
      "batch1", get_batch_lines(value, "batch1_lines")));
}

ModelConfig::ModelConfig(const vart::xrnn::RNNModelParser* mp)
    : model_parser_(mp), layers_(mp->get_num_layers()) {
  // Fill the layer_config_ with ext_mem reg config
  for (int i = 0; i < layers_; i++) {
    std::vector<LAYER_T> vl;
    vl.reserve(ext_mem_reg.size());
    for (const auto& reg_name : ext_mem_reg) {
      auto reg_data = mp->get_ddr_reg_config(i, reg_name);
      if (!reg_data.empty()) {
        LAYER_T lt = {static_cast<int>(reg_data[0]),       // direction
                      static_cast<int>(reg_data[1]) / 2};  // aligned size
        vl.push_back(lt);
      } else {
        LAYER_T lt = {0, 0};
        vl.push_back(lt);
      }
    }
    layer_config_.push_back(std::move(vl));
  }

  // Fill the batch_lines_ [Data in terms of #rows, not bytes]
  std::string name = "batch" + std::to_string(mp->get_batch_size());
  std::vector<int> batch_data;
  batch_data.reserve(layers_ * 2 + 1);
  for (int i = 0; i < layers_; ++i) {
    auto first = mp->get_first_instructions(i).size();  // in bytes
    auto loop = mp->get_loop_instructions(i).size();    // in bytes
    CHECK(first % 16 == 0);
    CHECK(loop % 16 == 0);
    batch_data.push_back(first / 16);
    batch_data.push_back(loop / 16);
  }
  auto end = mp->get_end_instructions().size();  // in bytes
  CHECK(end % 16 == 0);
  batch_data.push_back(end / 16);
  batch_lines_.insert(
      std::pair<std::string, std::vector<int>>(name, std::move(batch_data)));

  // Compose the full length instructions
  instrs_ = compose_instructions();
}

int ModelConfig::get_layer_num() const { return layers_; }

int ModelConfig::get_layer_instr_len(int layer_num, int batch) const {
  std::array<int, 5> batch_constants = {0, 0, 0, 0xb, 0xe};
  std::string batch_name = "batch" + std::to_string(batch);
  CHECK(batch_lines_.count(batch_name) == 1) << "Unsupport batch size!!!";
  const std::vector<int>& bl = batch_lines_.at(batch_name);
  int batch_constant = batch_constants[batch];

  return (bl[layer_num * 2] + bl[layer_num * 2 + 1] + batch_constant) * 0x10;
}

int ModelConfig::get_reg_size(int layer_num, CONFIG_NAME config) const {
  return layer_config_[layer_num][config].size;
}

int ModelConfig::get_reg_dir(int layer_num, CONFIG_NAME config) const {
  return layer_config_[layer_num][config].direction;
}

int ModelConfig::get_first_instr_count(int layer_num, int batch) const {
  std::string batch_name = "batch" + std::to_string(batch);
  CHECK(batch_lines_.count(batch_name) == 1) << "Unsupport batch size!!!";
  const std::vector<int>& bl = batch_lines_.at(batch_name);
  return bl[layer_num * 2];
}

int ModelConfig::get_loop_instr_count(int layer_num, int batch) const {
  std::string batch_name = "batch" + std::to_string(batch);
  CHECK(batch_lines_.count(batch_name) == 1) << "Unsupport batch size!!!";
  const std::vector<int>& bl = batch_lines_.at(batch_name);
  return bl[layer_num * 2 + 1];
}

int ModelConfig::get_end_instr_count(int batch) const {
  std::string batch_name = "batch" + std::to_string(batch);
  CHECK(batch_lines_.count(batch_name) == 1) << "Unsupport batch size!!!";
  const std::vector<int>& bl = batch_lines_.at(batch_name);
  return bl.back();
}

const std::vector<int>& ModelConfig::get_instr_count_vector(int batch) const {
  std::string batch_name = "batch" + std::to_string(batch);
  CHECK(batch_lines_.count(batch_name) == 1) << "Unsupport batch size!!!";
  return batch_lines_.at(batch_name);
}

std::vector<char> ModelConfig::compose_instructions() {
  const auto& target = model_parser_->get_target_device();
  std::vector<char> instructions;
  if (target == "U50") {
    generate_ddr_layout_u50();
    instructions = compose_instructions_u50();
    fill_ddr_config_u50(instructions);
  } else if (target == "U25") {
    generate_ddr_layout_u25();
    instructions = compose_instructions_u50();
    fill_ddr_config_u50(instructions);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
      << "Dumping ddr layout to ddr_layout.txt ... "
      << dump_ddr_instr_layout(ddr_instr_layout_, "ddr_layout.txt") << "Done";
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
      << "Dumping batch lines to ddr_batchlines.txt ... "
      << dump_batch_lines(batch_lines_, "ddr_batchlines.txt") << "Done";
  return instructions;
}

std::vector<char> ModelConfig::compose_instructions_u50() const {
  constexpr int instruction_width = 16;  // Bytes
  auto ninstrs = ddr_instr_layout_.back();
  std::vector<char> instructions(ninstrs * instruction_width);
  int nlayers = layers_;
  for (int i = 0; i < nlayers; ++i) {
    auto init_instr = model_parser_->get_first_instructions(i);
    auto loop_instr = model_parser_->get_loop_instructions(i);
    auto first_iter = std::next(
        instructions.begin(), ddr_instr_layout_[i * 3 + 1] * instruction_width);
    auto loop_iter = std::next(
        instructions.begin(), ddr_instr_layout_[i * 3 + 2] * instruction_width);
    std::copy(init_instr.begin(), init_instr.end(), first_iter);
    std::copy(loop_instr.begin(), loop_instr.end(), loop_iter);
  }
  auto end_instr = model_parser_->get_end_instructions();
  auto end_iter = std::next(instructions.begin(),
                            ddr_instr_layout_.end()[-2] * instruction_width);
  std::copy(end_instr.begin(), end_instr.end(), end_iter);
  return instructions;
}

std::vector<char> ModelConfig::compose_instructions_u25() const { return {}; }

std::pair<char*, size_t> ModelConfig::get_instructions() {
  return std::make_pair(instrs_.data(), instrs_.size());
}

void ModelConfig::fill_ddr_config_u50(std::vector<char>& instrns) const {
  // DDR bank offset for instructions
  constexpr uint32_t INSTR_OFFSET = 0x0700'0000;
  constexpr uint32_t BYTES_PER_ROW = 16;
  constexpr uint32_t REGS_PER_ROW = 4;

  // #Regs to skip in every header to reach first batch header
  constexpr uint32_t SKIP_REGS = 8;

  // Length of a batch header in terms of rows
  constexpr uint32_t BATCH_HEADER_ROWS = 3;
  constexpr uint32_t BATCH_HEADER_REGS = BATCH_HEADER_ROWS * REGS_PER_ROW;

  // Length of header per layer in terms of rows
  int batch = model_parser_->get_batch_size();
  const uint32_t LAYER_HEADER_LEN = BATCH_HEADER_ROWS * batch + 2;

  char* i8_base_ptr = instrns.data();
  uint32_t* i32_base_ptr = reinterpret_cast<uint32_t*>(i8_base_ptr);
  uint32_t* i32_ptr = i32_base_ptr;

  //
  // One-time Programming for a model
  //
  uint32_t num_end_instrs = get_end_instr_count(batch);
  uint32_t end_instr_offset = ddr_instr_layout_.end()[-2] * BYTES_PER_ROW;
  i32_ptr[4] = layers_;                          // num of layers
  i32_ptr[5] = INSTR_OFFSET + end_instr_offset;  // end_instr address
  i32_ptr[6] = num_end_instrs;                   // #end instrs (in rows)

  //
  // Programming DDR config header for all layers
  // Its values are constant for a compiled model
  //

  for (int i = 0; i < layers_; ++i) {
    i32_ptr = reinterpret_cast<uint32_t*>(i8_base_ptr) +
              ddr_instr_layout_[i * 3] * REGS_PER_ROW;
    uint32_t base_addr = INSTR_OFFSET;
    uint32_t cur_layer_header_len =
        (i == 0) ? LAYER_HEADER_LEN + 1 : LAYER_HEADER_LEN;
    uint32_t first_instr_offset = ddr_instr_layout_[i * 3 + 1] * BYTES_PER_ROW;
    uint32_t first_instr_count = get_first_instr_count(i, batch);
    uint32_t loop_instr_offset = ddr_instr_layout_[i * 3 + 2] * BYTES_PER_ROW;
    uint32_t loop_instr_count = get_loop_instr_count(i, batch);

    i32_ptr[HEAD_LEN_OFFSET] = cur_layer_header_len - 1;  // head-len
    constexpr uint32_t L0_SKIP = 4;  // Skip extra 4 reg for layer0
    if (i == 0) {
      i32_ptr[FINSTR_ADDR_OFFSET + L0_SKIP] = base_addr + first_instr_offset;
      i32_ptr[FINSTR_LEN_OFFSET + L0_SKIP] = first_instr_count;
      i32_ptr[LINSTR_ADDR_OFFSET + L0_SKIP] = base_addr + loop_instr_offset;
      i32_ptr[LINSTR_LEN_OFFSET + L0_SKIP] = loop_instr_count;
    } else {
      i32_ptr[FINSTR_ADDR_OFFSET] = base_addr + first_instr_offset;
      i32_ptr[FINSTR_LEN_OFFSET] = first_instr_count;
      i32_ptr[LINSTR_ADDR_OFFSET] = base_addr + loop_instr_offset;
      i32_ptr[LINSTR_LEN_OFFSET] = loop_instr_count;
    }

    // Store on-chip memory addresses generated by compiler
    uint32_t* bhead_ptr = i32_ptr + SKIP_REGS;
    if (i == 0) bhead_ptr += L0_SKIP;
    if (auto int_mem_reg_data =
            model_parser_->get_ddr_reg_config(i, "load_dst_reg_0");
        !int_mem_reg_data.empty()) {
      for (int b = 0; b < batch; ++b) {
        bhead_ptr[b * BATCH_HEADER_REGS + LOAD_DST0_OFFSET] =
            int_mem_reg_data.at(b);
      }
    }
    if (auto int_mem_reg_data =
            model_parser_->get_ddr_reg_config(i, "load_dst_reg_1");
        !int_mem_reg_data.empty()) {
      for (int b = 0; b < batch; ++b) {
        bhead_ptr[b * BATCH_HEADER_REGS + LOAD_DST1_OFFSET] =
            int_mem_reg_data.at(b);
      }
    }
  }
}

void ModelConfig::fill_ddr_config_u25(std::vector<char>& instructions) const {}

void ModelConfig::generate_ddr_layout_u25() {
  constexpr uint32_t ROW_ALIGNMENT = 16;
  const auto* mp = model_parser_;
  auto batch_size = mp->get_batch_size();
  auto header = batch_size * 3 + 2;

  uint32_t idx = 0;
  for (int i = 0; i < layers_; ++i) {
    auto head = (i == 0) ? header + 1 : header;
    auto first = mp->get_first_instructions(i).size();  // in bytes
    auto loop = mp->get_loop_instructions(i).size();    // in bytes
    CHECK(first % 16 == 0);
    CHECK(loop % 16 == 0);

    ddr_instr_layout_.push_back(idx);  // header index
    idx = align_to<ROW_ALIGNMENT>(idx + head);
    ddr_instr_layout_.push_back(idx);  // first_instr index
    idx = align_to<ROW_ALIGNMENT>(idx + first / 16);
    ddr_instr_layout_.push_back(idx);  // loop_instr index
    idx += (loop / 16);
  }

  idx = align_to<ROW_ALIGNMENT>(idx);
  ddr_instr_layout_.push_back(idx);  // end_instr index

  auto end = mp->get_end_instructions().size();  // in bytes
  CHECK(end % 16 == 0);
  idx = align_to<ROW_ALIGNMENT>(idx + end / 16);
  ddr_instr_layout_.push_back(idx);  // final row index
}

void ModelConfig::generate_ddr_layout_u50() {
  const auto* mp = model_parser_;
  auto batch_size = mp->get_batch_size();
  auto header = batch_size * 3 + 2;

  uint32_t idx = 0;
  for (int i = 0; i < layers_; ++i) {
    auto head = (i == 0) ? header + 1 : header;
    auto first = mp->get_first_instructions(i).size();  // in bytes
    auto loop = mp->get_loop_instructions(i).size();    // in bytes
    CHECK(first % 16 == 0);
    CHECK(loop % 16 == 0);

    ddr_instr_layout_.push_back(idx);  // header index
    idx += head;
    ddr_instr_layout_.push_back(idx);  // first_instr index
    idx += first / 16;
    ddr_instr_layout_.push_back(idx);  // loop_instr index
    idx += loop / 16;
  }

  ddr_instr_layout_.push_back(idx);              // end_instr index
  auto end = mp->get_end_instructions().size();  // in bytes
  CHECK(end % 16 == 0);
  idx += (end / 16);
  ddr_instr_layout_.push_back(idx);  // final row index
}

std::string ModelConfig::dump_instructions(const std::string& filename) const {
  LOG(INFO) << "Dumping instructions to " << filename;
  std::ofstream of(filename);
  if (!of.is_open()) {
    LOG(WARNING) << "Couldn't open " << filename;
    return "";
  }
  int nregs = instrs_.size() / 4;
  const uint32_t* ptr = reinterpret_cast<const uint32_t*>(instrs_.data());
  of << std::hex;
  for (int reg = 0; reg < nregs; reg += 4) {
    for (int i = 0; i < 4; ++i) {
      of << ptr[reg + i];
      if (i < 3) of << " ";
    }
    of << std::endl;
  }
  of << std::dec;
  return "";
}

const std::vector<uint32_t>& ModelConfig::get_ddr_layout() const {
  return ddr_instr_layout_;
}

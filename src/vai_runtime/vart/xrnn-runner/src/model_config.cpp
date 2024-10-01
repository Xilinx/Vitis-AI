/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <glog/logging.h>
#include "vitis/ai/env_config.hpp"
#include "model_config.hpp"

DEF_ENV_PARAM(DEBUG_MODEL_CONFIG, "0");

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

static struct LAYER_T get_layer_config(json_object * obj, const std::string& key)
{
  json_object* field = nullptr;
  CHECK(json_object_object_get_ex(obj, key.c_str(), &field))
    << "no such field! key=" << key << ", value="
    << json_object_to_json_string(obj);
  
  json_object* direction = nullptr;
  CHECK(json_object_object_get_ex(field, "dir", &direction)) 
    << "no such field! key=dir, value="
    << json_object_to_json_string(field);
  
  json_object* size = nullptr;
  CHECK(json_object_object_get_ex(field, "size", &size)) 
    << "no such field! key=dir, value="
    << json_object_to_json_string(field);

  struct LAYER_T l;
  l.direction = json_object_get_int(direction);
  l.size = json_object_get_int(size);

  return l;
}

static std::vector<int> get_batch_lines(json_object * obj, const std::string& key)
{
  json_object* field = nullptr;
  CHECK(json_object_object_get_ex(obj, key.c_str(), &field))
    << "no such field! key=" << key << ", value="
    << json_object_to_json_string(obj);

  auto ret = std::vector<int>{};
  auto size = json_object_array_length(field);
  ret.reserve(size);

  for(decltype(size) idx=0; idx<size; idx++){
    auto elt = json_object_array_get_idx(field, idx);
    CHECK(json_object_is_type(elt, json_type_int))
      << "element is not an int or array of int ! key=" << key
      << ", idx=" << idx << ", value=" << json_object_to_json_string(obj);
    ret.emplace_back(json_object_get_int(elt));
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_MODEL_CONFIG)) << ret.size();
  return ret;
}

ModelConfig::ModelConfig(const std::string & model_directory)
{
  auto value = read_json_from_directory(model_directory);

  json_object* field = nullptr;
  CHECK(json_object_object_get_ex(value, "layer", &field))
    << "no such field! key=layer, value="
    << json_object_to_json_string(value);

  layers_ = json_object_object_length(field);
  for(int i=0; i<layers_; i++){
    json_object* ith_layer = nullptr;

    CHECK(json_object_object_get_ex(field, std::to_string(i).c_str(), &ith_layer))
      << "no such field! key=" << i << ", value="
      << json_object_to_json_string(field);

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

  /*
  for(unsigned i=0; i<layer_config_.size(); i++){
    for(unsigned j=0; j<layer_config_[i].size(); j++){
      LOG_IF(INFO, ENV_PARAM(DEBUG_MODEL_CONFIG))
        << "dir: " << layer_config_[i][j].direction
        << " size: " << layer_config_[i][j].size;
  
    }
  }
  
  std::map<std::string, std::vector<int>> :: iterator it = batch_lines_.begin();
  for( it=batch_lines_.begin(); it!=batch_lines_.end(); ++it){
    LOG_IF(INFO, ENV_PARAM(DEBUG_MODEL_CONFIG)) << it->first;
    std::vector<int> b = it->second;
    for(unsigned i=0; i<b.size(); ++i){
      LOG_IF(INFO, ENV_PARAM(DEBUG_MODEL_CONFIG)) << b[i];
    }
  }
  */
}

ModelConfig::~ModelConfig() {

}

int ModelConfig::get_layer_num(void){
  return layers_;
}

int ModelConfig::get_layer_instr_len(int layer_num, int batch){

  std::string batch_name = "batch"+std::to_string(batch);
  CHECK(batch_lines_.count(batch_name)==1) << "Unsupport batch size!!!";
  std::vector<int>& bl = batch_lines_[batch_name];
  int batch_constant = batch==3?0xb:0xe;

  return (bl[layer_num*2] + bl[layer_num*2+1] + batch_constant)*0x10; 
}

int ModelConfig::get_reg_size(int layer_num, CONFIG_NAME config){
    return layer_config_[layer_num][config].size;
}

int ModelConfig::get_reg_dir(int layer_num, CONFIG_NAME config){
    return layer_config_[layer_num][config].direction;
}

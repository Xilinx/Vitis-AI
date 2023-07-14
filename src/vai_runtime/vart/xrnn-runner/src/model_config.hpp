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
#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <json-c/json.h>

struct LAYER_T
{
  int direction;
  int size;    
};

enum CONFIG_NAME
{
  LOAD0=0,
  LOAD1=1,
  SAVE0=2 
};

class ModelConfig{
public:
   
  explicit ModelConfig(const std::string& model_directory); 
  virtual ~ModelConfig();

public:

  int get_layer_num(void);
  int get_layer_instr_len(int layer_num, int batch);
  int get_reg_size(int layer_num, CONFIG_NAME config);
  int get_reg_dir(int layer_num, CONFIG_NAME config);
 
private:
  int layers_;
  std::vector<std::vector<LAYER_T>> layer_config_;
  std::map<std::string, std::vector<int>> batch_lines_;
};


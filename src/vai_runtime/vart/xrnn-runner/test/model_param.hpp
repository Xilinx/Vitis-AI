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
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>

#include "common.h"

static std::map<const std::string, const std::string> model_name_map{
  {"sent", "sentiment"},
  {"satis", "satisfaction"},
  {"oie", "openie"}
}; 

class ModelParameters{
public:

  explicit ModelParameters(std::string model_dir,
                        std::string model_name,
                        std::string str_frame)
  :model_dir_{model_dir}, model_name_{model_name}, str_frame_{str_frame}{
    vector_file_ = model_dir_ + "data/"+model_name_+str_frame_+"/vector_bin";
    bench_file_  = model_dir_ + "data/"+model_name_+str_frame_+"/bench_bin";
    xclbin_file_ = model_dir_ + "xclbin/u25_gen3x8/xrnn.hw.xclbin";
    ddr_file_ = model_dir_ + "data/" + model_name_ + "/ddr_bin";
    
    model_type_ = model_name_map.count(model_name_)==1?model_name_map[model_name_]:"";
    CHECK(model_type_!="");
    
    //LOG(INFO) << "model locate in " << model_dir_+ "data/"+model_name_+str_frame_;
  }
  virtual ~ModelParameters(){};

public:
  void set(int batch, int frames, int vin_len, int vout_len){
    idims_ = {batch, frames, vin_len, 2};
    odims_ = {batch, frames, vout_len, 2};
  };

  size_t get_batch_len(std::vector<int> dim){
    size_t batch_len = 1;
    for(unsigned i=1; i<dim.size(); i++){
        batch_len *= dim[i];
    }
    return batch_len;
  };

  std::string get_vector_file(){return vector_file_;};
  std::string get_bench_file(){return bench_file_;};
  std::string get_xclbin_file(){return xclbin_file_;};
  std::string get_ddr_file(){return ddr_file_;};
  std::string get_model_type(){return model_type_;};
  std::vector<int> get_input_dims(){return idims_;};
  std::vector<int> get_output_dims(){return odims_;};

  std::string get_model_folder(){
    return model_dir_+"data/"+model_name_+str_frame_+"/";
  };
  
  std::string get_model_root(){
    return model_dir_+"data/"+model_name_+"/";
  };

  void update_vector_name(std::string vector_name){
    vector_file_ = model_dir_+"data/"+model_name_+str_frame_+"/"+vector_name;
  };
  void update_bench_name(std::string bench_name){
    bench_file_ = model_dir_+"data/"+model_name_+str_frame_+"/"+bench_name;
  };
  void update_ddr_name(std::string ddr_name){
    ddr_file_ = model_dir_+"data/"+model_name_+"/"+ddr_name;
  };

private:
  std::string model_dir_;
  std::string model_name_;
  std::string str_frame_;

  std::string vector_file_;
  std::string bench_file_;
  std::string xclbin_file_;
  std::string ddr_file_;
  std::string model_type_;
  std::vector<int> idims_;
  std::vector<int> odims_;
};

class ModelTestData{
public:

  explicit ModelTestData(ModelParameters *mp){
    LOG(INFO) << mp->get_vector_file();
    std::tie(vector_data_, vector_size_) = read_binary_file(mp->get_vector_file());
    LOG(INFO) << mp->get_bench_file();
    std::tie(bench_data_, bench_size_) = read_binary_file(mp->get_bench_file());
    LOG(INFO) << mp->get_ddr_file();
    std::tie(model_data_, model_size_) = read_binary_file(mp->get_ddr_file());
  }
  virtual ~ModelTestData(){
    delete [] vector_data_;
    delete [] bench_data_;
    delete [] model_data_;
  };

public:
  char* get_data(std::string name){
    if(name == "vector")
      return vector_data_;
    else if(name == "bench")
      return bench_data_;
    else if(name == "ddr")
      return model_data_;
    else
      return NULL;
  };
  size_t get_size(std::string name){
    if(name == "vector")
      return vector_size_;
    else if(name == "bench")
      return bench_size_;
    else if(name == "ddr")
      return model_size_;
    else
      return 0;
  };

private:
  char*  vector_data_;
  size_t vector_size_;
  char*  bench_data_;
  size_t bench_size_;
  char*  model_data_;
  size_t model_size_;
};

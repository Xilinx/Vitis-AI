/*
 * Copyright 2021 xilinx Inc.
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

#include "multi_runner_imp.hpp"

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vart/batch_tensor_buffer_view.hpp>
#include <vector>
#include <vitis/ai/collection_helper.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/graph_runner.hpp>
#include <vitis/ai/path_util.hpp>
#include <vitis/ai/profiling.hpp>

DEF_ENV_PARAM_2(VAI_LIBRARY_MODELS_DIR, ".", std::string)
DEF_ENV_PARAM(DEBUG_MULTI_RUNNER, "0")
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0")

using namespace std;

namespace vitis {
namespace ai {

static vector<string> find_model_search_path() {
  auto ret = vector<string>{};
  ret.emplace_back(".");
  ret.emplace_back(ENV_PARAM(VAI_LIBRARY_MODELS_DIR));
  ret.emplace_back("/usr/share/vitis_ai_library/models");
  ret.emplace_back("/usr/share/vitis_ai_library/.models");
  return ret;
}

static size_t filesize(const string& filename) {
  size_t ret = 0u;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = S_ISREG(statbuf.st_mode) ? statbuf.st_size : 0u;
  }
  return ret;
}

string find_config_file(const string& name) {
  if (filesize(name) > 0u) {
    return name;
  }
  auto ret = std::string();
  for (const auto& p : find_model_search_path()) {
    ret = p + "/" + name + "/" + name;
    const auto config_file = ret + ".prototxt";
    if (filesize(config_file) > 0u) {
      return config_file;
    }
  }

  stringstream str;
  str << "cannot find prototxt <" << name << "> after checking following dir:";
  for (const auto& p : find_model_search_path()) {
    str << "\n\t" << p;
  }
  LOG(FATAL) << str.str();
  return string("");
}

static std::string slurp(const char* filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  CHECK(in.good()) << "failed to read config file. filename=" << filename;
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}

std::string to_string(const std::unique_ptr<subgraphParam>& node) {
  std::ostringstream str;
  str << "subgraph :" << node->subgraph->get_name() << " ";
  for (auto&& n : node->nexts) {
    str << " \n\t==> " << n->subgraph->get_name();
  }
  return str.str();
}

std::string to_string(const std::vector<std::unique_ptr<subgraphParam>>& list) {
  std::ostringstream str;
  for (auto&& l : list) {
    str << "\n" << to_string(l) << "\n";
  }
  return str.str();
}

static std::vector<std::string> split(const std::string& s,
                                      const std::string& delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.emplace_back(s.substr(pos, len - pos));
      break;
    }
    elems.emplace_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

MultiRunnerImp::MultiRunnerImp(std::string model_name)
    : attrs_{xir::Attrs::create()} {
  attrs_->set_attr("lib", std::map<std::string, std::string>{
                              {"CPU", "libvitis_ai_library-cpu_task.so.3"}});
  auto config_file = find_config_file(model_name);
  create_models(config_file);
  std::string pre_name = config_file.substr(0, config_file.rfind("/") + 1);
  create_graphs(pre_name);
  create_subgraphs();
  create_runner();
  create_tensor();
  tb_allocator_ = vart::assistant::TensorBufferAllocator::create(attrs_.get());
  create_tensor_buffers();
  link_tensor_buffers();
  // sort
  std::stable_sort(subgraphs_.begin(), subgraphs_.end(),
                   [](const auto& ls, const auto& rs) {
                     for (auto&& i : ls->nexts)
                       if (i == rs.get()) return true;
                     return false;
                   });
  LOG_IF(INFO, ENV_PARAM(DEBUG_MULTI_RUNNER)) << to_string(subgraphs_);
  create_out_tensors_tbs();
}
static void dump_tensor_buffer(const std::string& dir0,
                               vart::TensorBuffer* tensor_buffer) {
  auto maybe_remove_trail_slah = [](const std::string& s) {
    if (s.back() == '/') {
      return s.substr(0, s.size() - 1);
    }
    return s;
  };
  std::string dir = maybe_remove_trail_slah(dir0);
  vitis::ai::create_parent_path(dir);
  CHECK(vitis::ai::is_directory(dir)) << "cannot create directory: dir=" << dir;
  auto tensor_name = tensor_buffer->get_tensor()->get_name();
  auto tensor_name_remove_fix = xir::remove_xfix(tensor_name);
  auto filename0 = vitis::ai::to_valid_file_name(tensor_name_remove_fix);
  dir += "/" + filename0;
  std::vector<int> idx(tensor_buffer->get_tensor()->get_shape().size(), 0);
  auto batch = tensor_buffer->get_tensor()->get_shape()[0];
  auto datasize = tensor_buffer->get_tensor()->get_data_size() / batch;
  for (auto b = 0; b < batch; ++b) {
    idx[0] = b;
    uint64_t data = tensor_buffer->data(idx).first;
    auto filename = dir + "_" + std::to_string(b) + ".bin";
    CHECK(std::ofstream(filename).write((char*)data, datasize).good())
        << "failed to write: " << filename;
  }
  idx[0] = 0;
  uint64_t data = tensor_buffer->data(idx).first;
  auto filename = dir + ".bin";
  CHECK(std::ofstream(filename).write((char*)data, datasize * batch).good())
      << "failed to write: " << filename;
  return;
}

static void maybe_dump_tensor_buffers(const vitis::ai::subgraphParam* sub) {
  if (!ENV_PARAM(XLNX_ENABLE_DUMP)) {
    return;
  }
  auto sname = vitis::ai::to_valid_file_name(sub->subgraph->get_name());
  auto dir = "dump/" + std::to_string(sub->own_graph_idx) + sname + "/";
  for (auto&& i : sub->inputs) {
    auto dirname = dir + "i";
    dump_tensor_buffer(dirname, i.tensor_buffer.get());
  }
  for (auto&& i : sub->outputs) {
    auto dirname = dir + "o";
    dump_tensor_buffer(dirname, i.tensor_buffer.get());
  }
}

static void maybe_sync_for_read(vart::TensorBuffer* b) {
  switch (b->get_location()) {
    case vart::TensorBuffer::location_t::HOST_VIRT:
      // do nothing
      break;
    case vart::TensorBuffer::location_t::HOST_PHY:
      // TODO: check continous
      b->sync_for_read(0, b->get_tensor()->get_data_size());
      break;
    default:
      // update_proxy already copy the tensor buffer
      // do nothing LOG(FATAL) << "Not supported!";
      break;
  }
}

std::pair<uint32_t, int> MultiRunnerImp::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output_in) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_MULTI_RUNNER))
      << "MultiRunnerImp::execute_async";
  for (auto sub_idx = 0u; sub_idx < subgraphs_.size(); sub_idx++) {
    auto&& sub = subgraphs_[sub_idx];
    __TIC__(MultiRunner_execute_async)

    int runner_batch = sub->runner->get_input_tensors()[0]->get_shape()[0];

    auto view_tb = [](auto& internals, size_t batch_index, size_t batch) {
      return vitis::ai::vec_map(internals, [batch_index, batch](const auto& i) {
        return std::unique_ptr<vart::TensorBuffer>(
            new vart::BatchTensorBufferView(
                const_cast<vart::TensorBuffer*>(i.tensor_buffer.get()),
                batch_index, batch));
      });
    };
    int end_batch;
    for (auto i = 0; i < sub->cycles; i += end_batch) {
      end_batch = std::min(sub->cycles - i, runner_batch);
      auto input_ptr = view_tb(sub->inputs, i, end_batch);
      auto output_ptr = view_tb(sub->outputs, i, end_batch);
      auto inputs = vitis::ai::vector_unique_ptr_get(input_ptr);
      auto outputs = vitis::ai::vector_unique_ptr_get(output_ptr);
      auto status = sub->runner->execute_async(inputs, outputs);
      auto ok = sub->runner->wait((int)status.first, -1);
      CHECK(ok == 0);
    }
    for (auto& output : sub->outputs) {
      maybe_sync_for_read(output.tensor_buffer.get());
      auto linker = output.linker.get();
      if (linker) {
        linker->after_invoke_runner(sub->subgraph);
      }
    }
    maybe_dump_tensor_buffers(sub.get());
    LOG_IF(INFO, ENV_PARAM(DEEPHI_PROFILING))
        << "MultiRunner_execute_async " << sub_idx;
    __TOC__(MultiRunner_execute_async)
  }
  return std::make_pair(0u, 0);
}

void MultiRunnerImp::create_out_tensors_tbs() {
  for (auto&& sub : subgraphs_) {
    for (auto&& i : sub->inputs) {
      if (not_input_output_tensors_.count(i.my_tensor->get_name()) == 0) {
        input_tensors_.emplace_back(i.my_tensor.get());
        input_tensor_buffers_.emplace_back(i.tensor_buffer.get());
      }
    }
    for (auto&& i : sub->outputs) {
      if (not_input_output_tensors_.count(i.my_tensor->get_name()) == 0) {
        output_tensors_.emplace_back(i.my_tensor.get());
        output_tensor_buffers_.emplace_back(i.tensor_buffer.get());
      }
    }
  }
}

void MultiRunnerImp::link_tensor_buffers() {
  auto get_slaves = [&](subgraphParam* master, const Internal& output,
                        int graph_idx) {
    std::vector<std::unique_ptr<vart::TensorBuffer>*> ret;
    for (auto&& sub : subgraphs_) {
      for (auto& proto : sub->input_protos) {
        if (proto.pre_model_idx == graph_idx &&
            output.my_tensor->get_name().find(proto.pre_tensor_name) !=
                std::string::npos) {
          for (auto&& input : sub->inputs) {
            if (input.my_tensor->get_name().find(proto.name) !=
                std::string::npos) {
              CHECK_EQ(input.my_tensor->get_data_size(),
                       output.my_tensor->get_data_size());
              not_input_output_tensors_.emplace(input.my_tensor->get_name());
              ret.emplace_back(&input.tensor_buffer);
              master->nexts.emplace(sub.get());
              break;
            }
          }
          break;
        }
      }
    }
    return ret;
  };
  for (auto&& sub : subgraphs_) {
    for (auto&& output : sub->outputs) {
      auto master = &output.tensor_buffer;
      auto slaves = get_slaves(sub.get(), output, sub->own_graph_idx);
      if (slaves.empty()) {
        continue;
      }
      not_input_output_tensors_.emplace(output.my_tensor->get_name());
      output.linker = MUTensorBufferLinker::create(master);
      for (auto& t : slaves) {
        output.linker->add_slave(t);
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_MULTI_RUNNER))
          << "linker: " << output.linker->to_string();
      // output.linker->finalize();
    }
  }
}

void MultiRunnerImp::create_tensor_buffers() {
  auto create_view_vector_get = [](const std::vector<vart::TensorBuffer*>& from,
                                   int batch) {
    return vitis::ai::vec_map(from, [&](vart::TensorBuffer* x) {
      return std::unique_ptr<vart::TensorBuffer>(
          new vart::BatchTensorBufferView(x, 0, batch));
    });
  };

  auto internal_get_my_tensors = [](const auto& outputs) {
    return vitis::ai::vec_map(outputs, [](const auto& i) {
      return const_cast<const xir::Tensor*>(i.my_tensor.get());
    });
  };
  for (auto&& sub : subgraphs_) {
    std::vector<std::unique_ptr<vart::TensorBuffer>> input_tbs, output_tbs;
    auto r = dynamic_cast<vart::RunnerExt*>(sub->runner.get());
    if (r && sub->inputs[0].runner_tensor->get_shape()[0] == sub->cycles) {
      input_tbs = create_view_vector_get(r->get_inputs(), sub->cycles);
      output_tbs = create_view_vector_get(r->get_outputs(), sub->cycles);
    } else {
      attrs_->set_attr<size_t>("__batch__", sub->cycles);
      attrs_->set_attr<int>(
          sub->subgraph->get_name() + ":__tensor_buffer_location__",
          (int)vart::TensorBuffer::location_t::HOST_VIRT);
      std::tie(input_tbs, output_tbs) = tb_allocator_->allocate(
          sub->subgraph, internal_get_my_tensors(sub->inputs),
          internal_get_my_tensors(sub->outputs));
    }
    for (size_t i = 0; i < sub->inputs.size(); i++) {
      sub->inputs[i].tensor_buffer = std::move(input_tbs[i]);
    }
    for (size_t i = 0; i < sub->outputs.size(); i++) {
      sub->outputs[i].tensor_buffer = std::move(output_tbs[i]);
    }
  }
}

void MultiRunnerImp::create_tensor() {
  auto copy_new_tensor = [](const xir::Tensor* t, int new_batch) {
    auto shape = t->get_shape();
    CHECK_GT(shape.size(), 0);
    shape[0] = new_batch;
    auto ret = xir::Tensor::create(t->get_name(), shape, t->get_data_type());
    ret->set_attrs(t->get_attrs());
    return ret;
  };

  for (auto&& sub : subgraphs_) {
    auto inputs = sub->runner->get_input_tensors();
    sub->inputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      sub->inputs[i].runner_tensor = inputs[i];
      sub->inputs[i].my_tensor = copy_new_tensor(inputs[i], sub->cycles);
    }
    auto outputs = sub->runner->get_output_tensors();
    sub->outputs.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      sub->outputs[i].runner_tensor = outputs[i];
      sub->outputs[i].my_tensor = copy_new_tensor(outputs[i], sub->cycles);
    }
  }
}
void MultiRunnerImp::create_runner() {
  for (auto&& i : subgraphs_) {
    if (i->subgraph->is_root()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_MULTI_RUNNER))
          << "create graph runner" << i->subgraph->get_name();
      i->runner = GraphRunner::create_graph_runner(i->subgraph->get_graph(),
                                                   attrs_.get());
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_MULTI_RUNNER))
          << "create vart runner" << i->subgraph->get_name();
      i->runner =
          vart::Runner::create_runner_with_attrs(i->subgraph, attrs_.get());
    }
  }
}
void MultiRunnerImp::create_graphs(const std::string& pre_name) {
  for (size_t j = 0; j < models_.size(); j++) {
    auto model_name = pre_name + models_[j].name() + ".xmodel";
    if (filesize(model_name) == 0u) {
      LOG(ERROR) << "connot find " << model_name;
    }
    graphs_.emplace_back(xir::Graph::deserialize(model_name));
  }
}
void MultiRunnerImp::create_subgraphs() {
  for (size_t k = 0; k < models_.size(); k++) {
    for (int i = 0; i < models_[k].subgraph_size(); i++) {
      auto sub = std::unique_ptr<subgraphParam>(new subgraphParam());
      sub->own_graph_idx = k;
      const auto& msub = models_[k].subgraph(i);
      sub->cycles = msub.cycles();
      std::vector<uint32_t> subgraph_idx;
      auto sub_strs = split(msub.subgraph_idx(), ".");
      std::transform(sub_strs.begin(), sub_strs.end(),
                     std::back_inserter(subgraph_idx),
                     [](std::string s) { return std::stoi(s); });
      sub->subgraph = graphs_[k]->get_root_subgraph();
      CHECK_GT(subgraph_idx.size(), 0);
      CHECK_EQ(subgraph_idx[0], 0);
      for (size_t j = 1; j < subgraph_idx.size(); j++) {
        auto childs = sub->subgraph->children_topological_sort();
        CHECK_LT(subgraph_idx[j], childs.size());
        sub->subgraph = childs[subgraph_idx[j]];
      }

      for (int j = 0; j < msub.input_size(); j++) {
        const auto& input = msub.input(j);
        sub->input_protos.push_back(
            InputPrototxt{input.name(), input.previous_model_idx(),
                          input.previous_output_tensor()});
      }
      subgraphs_.emplace_back(std::move(sub));
    }
  }
}

void MultiRunnerImp::create_models(const std::string& config_file) {
  vitis::ai::proto::DpuModelParamList mlist;
  auto text = slurp(config_file.c_str());
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &mlist);
  CHECK(ok) << "cannot parse config file. config_file=" << config_file;
  for (int i = 0; i < mlist.model_size(); i++) {
    models_.emplace_back(mlist.model(i));
  }
}

std::vector<const xir::Tensor*> MultiRunnerImp::get_input_tensors() {
  return input_tensors_;
}

std::vector<const xir::Tensor*> MultiRunnerImp::get_output_tensors() {
  return output_tensors_;
}

std::vector<vart::TensorBuffer*> MultiRunnerImp::get_inputs() {
  return input_tensor_buffers_;
}

std::vector<vart::TensorBuffer*> MultiRunnerImp::get_outputs() {
  return output_tensor_buffers_;
}

int MultiRunnerImp::wait(int jobid, int timeout) { return 0; }

std::vector<float> MultiRunnerImp::getMean() {
  CHECK_GT(models_[0].kernel_size(), 0);
  return std::vector<float>(models_[0].kernel(0).mean().begin(),
                            models_[0].kernel(0).mean().end());
}

std::vector<float> MultiRunnerImp::getScale() {
  CHECK_GT(models_[0].kernel_size(), 0);
  return std::vector<float>(models_[0].kernel(0).scale().begin(),
                            models_[0].kernel(0).scale().end());
}

}  // namespace ai
}  // namespace vitis

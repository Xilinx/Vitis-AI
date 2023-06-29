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

#include "vart/runner_helper.hpp"

#include <sstream>
#include <xir/tensor/tensor.hpp>
#include <xir/util/tool_function.hpp>

#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/path_util.hpp"
DEF_ENV_PARAM(DEBUG_RUNNER_HELPER, "0")
using std::to_string;
static std::string to_string(const std::pair<void*, size_t>& v) {
  std::ostringstream str;
  str << "@(" << v.first << "," << std::dec << v.second << ")";
  return str.str();
}

static std::string to_string(const std::pair<uint64_t, size_t>& v) {
  std::ostringstream str;
  str << "@(0x" << std::hex << v.first << "," << std::dec << v.second << ")";
  return str.str();
}

template <typename T>
std::string to_string(T begin, T end, char s = '[', char e = ']',
                      char sep = ',') {
  std::ostringstream str;
  str << s;
  int c = 0;
  for (auto it = begin; it != end; ++it) {
    if (c++ != 0) {
      str << sep;
    };
    str << to_string(*it);
  }
  str << e;
  return str.str();
}

std::string to_string(const vitis::ai::TensorBuffer* tensor_buffer) {
  auto dims = tensor_buffer->get_tensor()->get_dims();
  auto idx = dims;
  std::fill(idx.begin(), idx.end(), 0u);
  auto batch_size = dims[0];
  auto data_size = std::vector<std::pair<void*, size_t>>(batch_size);
  for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    idx[0] = batch_idx;
    data_size[batch_idx] =
        const_cast<vitis::ai::TensorBuffer*>(tensor_buffer)->data(idx);
  }
  std::ostringstream str;
  str << "TensorBuffer@" << (void*)tensor_buffer
      << "{data=" << to_string(data_size.begin(), data_size.end())
      << ", tensor=" << to_string(tensor_buffer->get_tensor()) << "}";
  return str.str();
}

std::string to_string(const vart::TensorBuffer* tensor_buffer) {
  auto dims = tensor_buffer->get_tensor()->get_shape();
  auto idx = dims;
  std::fill(idx.begin(), idx.end(), 0u);
  auto batch_size = dims[0];
  auto data_size = std::vector<std::pair<uint64_t, size_t>>(batch_size);
  for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    idx[0] = batch_idx;
    data_size[batch_idx] =
        const_cast<vart::TensorBuffer*>(tensor_buffer)->data(idx);
  }
  std::ostringstream str;
  str << "TensorBuffer@" << (void*)tensor_buffer
      << "{data=" << to_string(data_size.begin(), data_size.end())
      << ", tensor=" << to_string(tensor_buffer->get_tensor()) << "}";
  return str.str();
}

std::string to_string(const xir::Tensor* tensor) {
  std::ostringstream str;
  auto dims = tensor->get_shape();
  str << "Tensor@" << (void*)tensor << "{"                //
      << "name=" << tensor->get_name()                    //
      << ",dims=" << to_string(dims.begin(), dims.end())  //
      << "}"                                              //
      ;
  return str.str();
}

std::string to_string(const vitis::ai::Tensor* tensor) {
  std::ostringstream str;
  auto dims = tensor->get_dims();
  str << "Tensor@" << (void*)tensor << "{"                //
      << "name=" << tensor->get_name()                    //
      << ",dims=" << to_string(dims.begin(), dims.end())  //
      << "}"                                              //
      ;
  return str.str();
}

std::string to_string(const std::vector<vart::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}

std::string to_string(
    const std::vector<vitis::ai::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}
std::string to_string(const std::vector<xir::Tensor*>& tensors) {
  return to_string(tensors.begin(), tensors.end());
}
std::string to_string(
    const std::vector<const vitis::ai::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}
std::string to_string(
    const std::vector<const vart::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}
std::string to_string(const std::vector<const xir::Tensor*>& tensors) {
  return to_string(tensors.begin(), tensors.end());
}
namespace vitis {
namespace ai {
std::vector<std::unique_ptr<vitis::ai::TensorBuffer>>
alloc_cpu_flat_tensor_buffers(const std::vector<vitis::ai::Tensor*>& tensors) {
  auto ret =
      std::vector<std::unique_ptr<vitis::ai::TensorBuffer>>(tensors.size());
  for (auto i = 0u; i < tensors.size(); ++i) {
    ret[i] = std::unique_ptr<vitis::ai::TensorBuffer>(
        new vitis::ai::CpuFlatTensorBufferOwned(tensors[i]));
  }
  return ret;
}
}  // namespace ai
}  // namespace vitis

namespace vart {
std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}

tensor_buffer_data_t get_tensor_buffer_data(vart::TensorBuffer* tensor_buffer,
                                            size_t batch_index) {
  tensor_buffer_data_t ret;
  auto idx = vart::get_index_zeros(tensor_buffer->get_tensor());
  idx[0] = (int)batch_index;
  uint64_t data;
  std::tie(data, ret.size) = tensor_buffer->data(idx);
  ret.data = (void*)data;
  return ret;
}

tensor_buffer_data_t get_tensor_buffer_data(vart::TensorBuffer* tensor_buffer,
                                            const std::vector<int>& idx) {
  tensor_buffer_data_t ret;
  uint64_t data;
  std::tie(data, ret.size) = tensor_buffer->data(idx);
  ret.data = (void*)data;
  return ret;
}

class CpuFlatTensorBuffer : public TensorBuffer {
 public:
  explicit CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor);
  virtual ~CpuFlatTensorBuffer() = default;

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override;

 protected:
  void* data_;
  // std::unique_ptr<xir::Tensor> tensor_;
};

class CpuFlatTensorBufferOwned : public CpuFlatTensorBuffer {
 public:
  explicit CpuFlatTensorBufferOwned(const xir::Tensor* tensor);
  virtual ~CpuFlatTensorBufferOwned() = default;

 private:
  std::vector<char> buffer_;
};

CpuFlatTensorBuffer::CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
    : TensorBuffer{tensor}, data_{data} {}

std::pair<uint64_t, size_t> CpuFlatTensorBuffer::data(
    const std::vector<int> idx) {
  if (idx.size() == 0) {
    return {reinterpret_cast<uint64_t>(data_), tensor_->get_data_size()};
  }
  auto dims = tensor_->get_shape();
  auto offset = 0;
  for (auto k = 0u; k < dims.size(); k++) {
    auto stride = 1;
    for (auto m = k + 1; m < dims.size(); m++) {
      stride *= dims[m];
    }
    offset += idx[k] * stride;
  }

  auto dtype_size = tensor_->get_data_type().bit_width / 8;
  auto elem_num = tensor_->get_element_num();

  return std::make_pair(reinterpret_cast<uint64_t>(data_) + offset * dtype_size,
                        (elem_num - offset) * dtype_size);
}

CpuFlatTensorBufferOwned::CpuFlatTensorBufferOwned(const xir::Tensor* tensor)
    : CpuFlatTensorBuffer(nullptr, tensor), buffer_(tensor_->get_data_size()) {
  data_ = (void*)&buffer_[0];
}

std::vector<std::unique_ptr<vart::TensorBuffer>> alloc_cpu_flat_tensor_buffers(
    const std::vector<const xir::Tensor*>& tensors) {
  return vitis::ai::vec_map(tensors, alloc_cpu_flat_tensor_buffer);
}

std::unique_ptr<vart::TensorBuffer> alloc_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor) {
  return std::unique_ptr<vart::TensorBuffer>(
      new CpuFlatTensorBufferOwned(tensor));
}

void dump_tensor_buffer(const std::string& dir0,
                        vart::TensorBuffer* tensor_buffer, int batch_base) {
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
  auto idx = get_index_zeros(tensor_buffer->get_tensor());
  auto batch = tensor_buffer->get_tensor()->get_shape()[0];
  if (batch < 32) {
    for (auto b = 0; b < batch; ++b) {
      auto data = get_tensor_buffer_data(tensor_buffer, idx);
      tensor_buffer->sync_for_read(0u, data.size);
      auto filename =
          dir + "/" + filename0 + "_" + std::to_string(b + batch_base) + ".bin";
      CHECK(std::ofstream(filename)
                .write((char*)data.data, data.size / batch)
                .good())
          << "failed to write: " << filename;
      LOG_IF(INFO, true || ENV_PARAM(DEBUG_RUNNER_HELPER))
          << "write tensor buffer @" << (void*)tensor_buffer << ":"
          << " tensor_name=" << tensor_name  //
          << " filename=" << filename        //
          << " data=" << data.data           //
          << " size=" << data.size / batch;
      idx[0] = b;
    }
  } else {
    auto data = get_tensor_buffer_data(tensor_buffer, idx);
    tensor_buffer->sync_for_read(0u, data.size);
    auto filename =
        dir + "/" + filename0 + "_" + std::to_string(batch_base) + ".bin";
    CHECK(std::ofstream(filename).write((char*)data.data, data.size).good())
        << "failed to write: " << filename;
    LOG_IF(INFO, true || ENV_PARAM(DEBUG_RUNNER_HELPER))
        << "write tensor buffer @" << (void*)tensor_buffer << ":"
        << " tensor_name=" << tensor_name  //
        << " filename=" << filename        //
        << " data=" << data.data           //
        << " size=" << data.size;
  }
  return;
}
}  // namespace vart

namespace xir {
template <typename... T>
struct SupportedAttryTypes {
  using types = std::tuple<T...>;
};

using bytes_t = std::vector<int8_t>;

using ListOfSupportedAttryType1 = SupportedAttryTypes<
    /** begin supported list of attr types */
    bool,                                             //
    int8_t,                                           //
    uint8_t,                                          //
    int16_t, uint16_t,                                //
    int32_t, uint32_t,                                //
    int64_t, uint64_t,                                //
    float, double,                                    //
    std::string,                                      //
    bytes_t,                                          //
    std::vector<bool>,                                //
    std::vector<int8_t>, std::vector<uint8_t>,        //
    std::vector<int16_t>, std::vector<uint16_t>,      //
    std::vector<int32_t>, std::vector<uint32_t>,      //
    std::vector<int64_t>, std::vector<uint64_t>,      //
    std::vector<float>, std::vector<double>,          //
    std::vector<std::string>,                         //
    std::vector<bytes_t>,                             //
    std::map<std::string, int8_t>,                    //
    std::map<std::string, uint8_t>,                   //
    std::map<std::string, int16_t>,                   //
    std::map<std::string, uint16_t>,                  //
    std::map<std::string, int32_t>,                   //
    std::map<std::string, uint32_t>,                  //
    std::map<std::string, int64_t>,                   //
    std::map<std::string, uint64_t>,                  //
    std::map<std::string, float>,                     //
    std::map<std::string, double>,                    //
    std::map<std::string, std::string>,               //
    std::map<std::string, bytes_t>,                   //
    std::map<std::string, std::vector<bool>>,         //
    std::map<std::string, std::vector<int8_t>>,       //
    std::map<std::string, std::vector<uint8_t>>,      //
    std::map<std::string, std::vector<int16_t>>,      //
    std::map<std::string, std::vector<uint16_t>>,     //
    std::map<std::string, std::vector<int32_t>>,      //
    std::map<std::string, std::vector<uint32_t>>,     //
    std::map<std::string, std::vector<int64_t>>,      //
    std::map<std::string, std::vector<uint64_t>>,     //
    std::map<std::string, std::vector<float>>,        //
    std::map<std::string, std::vector<double>>,       //
    std::map<std::string, std::vector<std::string>>,  //
    std::map<std::string, std::vector<bytes_t>>,      //
    nullptr_t>;

using ListOfSupportedAttryType =
    SupportedAttryTypes<std::string,                        //
                        std::map<std::string, std::string>  //
                        >;

template <typename Op>
struct Apply {
  std::string do_it(const std::any& x) {
    return do_it(ListOfSupportedAttryType(), x);
  }
  template <typename T0, typename... T>
  std::string do_it(SupportedAttryTypes<T0, T...> tag, const std::any& x) {
    if (x.type() == typeid(T0)) {
      // suppress coverity complain
      try{
        return Op()(std::any_cast<T0>(x));
      } catch (std::exception& e){
        std::cerr <<"Should never run here with exception " << e.what() <<"\n";
        abort();
      }
    }
    return do_it(SupportedAttryTypes<T...>(), x);
  }
  std::string do_it(SupportedAttryTypes<> tag, const std::any& x) {
    return "NA";
  };
};

template <typename T, class = void>
struct is_cout_able : public std::false_type {};
template <typename T>
struct is_cout_able<
    T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : public std::true_type {};

template <typename T>
constexpr auto is_cout_able_v = is_cout_able<T>::value;

template <typename K, typename V>
std::enable_if_t<is_cout_able_v<K> && is_cout_able_v<V>, std::ostream&>
operator<<(std::ostream& out, const std::map<K, V>& v) {
  out << "{";
  for (const auto& x : v) {
    out << "\n\t\"" << x.first << "\" = " << x.second;
  }
  out << "\n}";
  return out;
}

struct ToString {
  template <typename T>
  std::enable_if_t<is_cout_able_v<T>, std::string> operator()(const T& x) {
    std::ostringstream str;
    str << x;
    return str.str();
  }
  template <typename T>
  std::enable_if_t<!is_cout_able_v<T>, std::string> operator()(const T& x) {
    std::ostringstream str;
    str << "unknwon type: " << typeid(x).name() << " and " << is_cout_able_v<T>;
    return str.str();
  }
};

std::string to_string(const xir::Attrs* attr) {
  std::ostringstream str;
  for (auto& key : attr->get_keys()) {
    str << '"' << key
        << "\" = " << Apply<ToString>().do_it(attr->get_attr(key));
    str << std::endl;
  }
  return str.str();
}
std::string to_string(const xir::OpDef* opdef) {
  std::ostringstream str;
  str << "OpDef{"                                 //
      << "name=" << opdef->name()                 //
      << ",#args=" << opdef->input_args().size()  //
      << to_string(opdef->input_args())           //
      << "\n\tdoc:" << opdef->annotation()        //
      << "}";                                     //
  return str.str();
}
std::string to_string(const std::vector<OpArgDef>& argdef) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto& arg : argdef) {
    if (c != 0) {
      str << ",";
    }
    str << to_string(arg);
    c++;
  }
  str << "]";
  return str.str();
}

std::string to_string(const OpArgDef& argdef) {
  std::ostringstream str;
  str << "argname: " << argdef.name;
  switch (argdef.occur_type) {
    case OpArgDef::REQUIRED:
      str << " (required)";
      break;
    case OpArgDef::OPTIONAL:
      str << " (optional)";
      break;
    case OpArgDef::REPEATED:
      str << " (repeated)";
      break;
    case OpArgDef::REQUIRED_AND_REPEATED:
      str << " (required/repeated)";
      break;
    default:
      str << "()";
      break;
  }
  // str << ":: " << argdef.data_type.to_string();
  str << "// " << argdef.annotation;
  return str.str();
}

}  // namespace xir

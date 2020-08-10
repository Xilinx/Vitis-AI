/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#ifdef GEMMLOWP_PROFILING
#include "profiling/profiler.h"
#endif

#ifdef TFLITE_CUSTOM_OPS_HEADER
void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);
#endif

namespace tflite {
namespace benchmark {
namespace {

// Backward compat with previous approach to enabling op profiling.
#if defined(TFLITE_PROFILING_ENABLED)
constexpr int kOpProfilingEnabledDefault = true;
#else
constexpr int kOpProfilingEnabledDefault = false;
#endif

// Dumps profiling events if profiling is enabled.
class ProfilingListener : public BenchmarkListener {
 public:
  explicit ProfilingListener(Interpreter* interpreter, uint32_t max_num_entries)
      : interpreter_(interpreter), profiler_(max_num_entries) {
    TFLITE_BENCHMARK_CHECK(interpreter);
    interpreter_->SetProfiler(&profiler_);
  }

  void OnSingleRunStart(RunType run_type) override;

  void OnSingleRunEnd() override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  Interpreter* interpreter_;
  profiling::BufferedProfiler profiler_;
  profiling::ProfileSummarizer summarizer_;
};

// Dumps gemmlowp profiling events if gemmlowp profiling is enabled.
class GemmlowpProfilingListener : public BenchmarkListener {
 public:
  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;
};

void ProfilingListener::OnSingleRunStart(RunType run_type) {
  if (run_type == REGULAR) {
    profiler_.Reset();
    profiler_.StartProfiling();
  }
}

void ProfilingListener::OnBenchmarkEnd(const BenchmarkResults& results) {
  if (summarizer_.HasProfiles()) {
    TFLITE_LOG(INFO) << summarizer_.GetOutputString();
  }
}

void ProfilingListener::OnSingleRunEnd() {
  profiler_.StopProfiling();
  auto profile_events = profiler_.GetProfileEvents();
  summarizer_.ProcessProfiles(profile_events, *interpreter_);
}

void GemmlowpProfilingListener::OnBenchmarkStart(
    const BenchmarkParams& params) {
#ifdef GEMMLOWP_PROFILING
  gemmlowp::RegisterCurrentThreadForProfiling();
  gemmlowp::StartProfiling();
#endif
}

void GemmlowpProfilingListener::OnBenchmarkEnd(
    const BenchmarkResults& results) {
#ifdef GEMMLOWP_PROFILING
  gemmlowp::FinishProfiling();
#endif
}

std::vector<std::string> Split(const std::string& str, const char delim) {
  std::vector<std::string> results;
  if (!util::SplitAndParse(str, delim, &results)) {
    results.clear();
  }
  return results;
}

template <typename T>
void FillRandomValue(T* ptr, int num_elements,
                     const std::function<T()>& random_func) {
  for (int i = 0; i < num_elements; ++i) {
    *ptr++ = random_func();
  }
}

void FillRandomString(tflite::DynamicBuffer* buffer,
                      const std::vector<int>& sizes,
                      const std::function<string()>& random_func) {
  int num_elements = 1;
  for (int dim : sizes) {
    num_elements *= dim;
  }
  for (int i = 0; i < num_elements; ++i) {
    auto str = random_func();
    buffer->AddString(str.data(), str.length());
  }
}

TfLiteStatus PopulateInputLayerInfo(
    const string& names_string, const string& shapes_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
  info->clear();
  std::vector<std::string> names = Split(names_string, ',');
  std::vector<std::string> shapes = Split(shapes_string, ':');

  if (names.size() != shapes.size()) {
    TFLITE_LOG(ERROR) << "The number of items in"
                      << " --input_layer_shape (" << shapes_string << ", with "
                      << shapes.size() << " items)"
                      << " must match the number of items in"
                      << " --input_layer (" << names_string << ", with "
                      << names.size() << " items)."
                      << " For example --input_layer=input1,input2"
                      << " --input_layer_shape=1,224,224,4:1,20";
    return kTfLiteError;
  }

  for (int i = 0; i < names.size(); ++i) {
    info->push_back(BenchmarkTfLiteModel::InputLayerInfo());
    BenchmarkTfLiteModel::InputLayerInfo& input = info->back();

    input.name = names[i];

    TFLITE_BENCHMARK_CHECK(util::SplitAndParse(shapes[i], ',', &input.shape))
        << "Incorrect size string specified: " << shapes[i];
    for (int dim : input.shape) {
      if (dim == -1) {
        TFLITE_LOG(ERROR)
            << "Any unknown sizes in the shapes (-1's) must be replaced"
            << " with the size you want to benchmark with.";
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

std::vector<int> TfLiteIntArrayToVector(const TfLiteIntArray* int_array) {
  std::vector<int> values;
  values.reserve(int_array->size);
  for (size_t i = 0; i < int_array->size; i++) {
    values.push_back(int_array->data[i]);
  }
  return values;
}

}  // namespace

BenchmarkParams BenchmarkTfLiteModel::DefaultParams() {
  BenchmarkParams default_params = BenchmarkModel::DefaultParams();
  default_params.AddParam("graph", BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_shape",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("use_nnapi", BenchmarkParam::Create<bool>(false));
  default_params.AddParam("use_legacy_nnapi",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam("nnapi_accelerator_name",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("use_gpu", BenchmarkParam::Create<bool>(false));
#if defined(__ANDROID__)
  default_params.AddParam("gpu_precision_loss_allowed",
                          BenchmarkParam::Create<bool>(true));
  default_params.AddParam(
      "gpu_gl_object_type",
      BenchmarkParam::Create<int32_t>(TFLITE_GL_OBJECT_TYPE_FASTEST));
#endif
  default_params.AddParam("allow_fp16", BenchmarkParam::Create<bool>(false));
  default_params.AddParam("require_full_delegation",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam(
      "enable_op_profiling",
      BenchmarkParam::Create<bool>(kOpProfilingEnabledDefault));
  default_params.AddParam("max_profiling_buffer_entries",
                          BenchmarkParam::Create<int32_t>(1024));
  return default_params;
}

BenchmarkTfLiteModel::BenchmarkTfLiteModel()
    : BenchmarkTfLiteModel(DefaultParams()) {}

BenchmarkTfLiteModel::BenchmarkTfLiteModel(BenchmarkParams params)
    : BenchmarkModel(std::move(params)) {}

void BenchmarkTfLiteModel::CleanUp() {
  if (inputs_data_.empty()) {
    return;
  }
  // Free up any pre-allocated tensor data during PrepareInputData.
  for (int i = 0; i < inputs_data_.size(); ++i) {
    delete[] inputs_data_[i].data.raw;
  }
  inputs_data_.clear();
}

BenchmarkTfLiteModel::~BenchmarkTfLiteModel() { CleanUp(); }

std::vector<Flag> BenchmarkTfLiteModel::GetFlags() {
  std::vector<Flag> flags = BenchmarkTfLiteModel::BenchmarkModel::GetFlags();
  std::vector<Flag> specific_flags = {
    CreateFlag<std::string>("graph", &params_, "graph file name"),
    CreateFlag<std::string>("input_layer", &params_, "input layer names"),
    CreateFlag<std::string>("input_layer_shape", &params_, "input layer shape"),
    CreateFlag<bool>("use_nnapi", &params_, "use nnapi delegate api"),
    CreateFlag<bool>("use_legacy_nnapi", &params_, "use legacy nnapi api"),
    CreateFlag<std::string>(
        "nnapi_accelerator_name", &params_,
        "the name of the nnapi accelerator to use (requires Android Q+)"),
    CreateFlag<bool>("use_gpu", &params_, "use gpu"),
#if defined(__ANDROID__)
    CreateFlag<bool>("gpu_precision_loss_allowed", &params_,
                     "Allow to process computation in lower precision than "
                     "FP32 in GPU. By default, it's enabled."),
    CreateFlag<int32_t>("gpu_gl_object_type", &params_,
                        "The preferred GL object type to represent tensors in "
                        "GPU. By default, it's TFLITE_GL_OBJECT_TYPE_FASTEST"),
#endif
    CreateFlag<bool>("allow_fp16", &params_, "allow fp16"),
    CreateFlag<bool>("require_full_delegation", &params_,
                     "require delegate to run the entire graph"),
    CreateFlag<bool>("enable_op_profiling", &params_, "enable op profiling"),
    CreateFlag<int32_t>("max_profiling_buffer_entries", &params_,
                        "max profiling buffer entries")
  };

  flags.insert(flags.end(), specific_flags.begin(), specific_flags.end());
  return flags;
}

void BenchmarkTfLiteModel::LogParams() {
  BenchmarkModel::LogParams();
  TFLITE_LOG(INFO) << "Graph: [" << params_.Get<std::string>("graph") << "]";
  TFLITE_LOG(INFO) << "Input layers: ["
                   << params_.Get<std::string>("input_layer") << "]";
  TFLITE_LOG(INFO) << "Input shapes: ["
                   << params_.Get<std::string>("input_layer_shape") << "]";
  TFLITE_LOG(INFO) << "Use nnapi : [" << params_.Get<bool>("use_nnapi") << "]";
  TFLITE_LOG(INFO) << "Use legacy nnapi : ["
                   << params_.Get<bool>("use_legacy_nnapi") << "]";
  if (!params_.Get<std::string>("nnapi_accelerator_name").empty()) {
    TFLITE_LOG(INFO) << "nnapi accelerator name: ["
                     << params_.Get<string>("nnapi_accelerator_name") << "]";
  }
  TFLITE_LOG(INFO) << "Use gpu : [" << params_.Get<bool>("use_gpu") << "]";
#if defined(__ANDROID__)
  TFLITE_LOG(INFO) << "Allow lower precision in gpu : ["
                   << params_.Get<bool>("gpu_precision_loss_allowed") << "]";
  TFLITE_LOG(INFO) << "Preferred GL object type in gpu : ["
                   << params_.Get<int32_t>("gpu_gl_object_type") << "]";
#endif
  TFLITE_LOG(INFO) << "Allow fp16 : [" << params_.Get<bool>("allow_fp16")
                   << "]";
  TFLITE_LOG(INFO) << "Require full delegation : ["
                   << params_.Get<bool>("require_full_delegation") << "]";
  TFLITE_LOG(INFO) << "Enable op profiling: ["
                   << params_.Get<bool>("enable_op_profiling") << "]";
  TFLITE_LOG(INFO) << "Max profiling buffer entries: ["
                   << params_.Get<int32_t>("max_profiling_buffer_entries")
                   << "]";
}

TfLiteStatus BenchmarkTfLiteModel::ValidateParams() {
  if (params_.Get<std::string>("graph").empty()) {
    TFLITE_LOG(ERROR)
        << "Please specify the name of your TF Lite input file with --graph";
    return kTfLiteError;
  }
  return PopulateInputLayerInfo(params_.Get<std::string>("input_layer"),
                                params_.Get<std::string>("input_layer_shape"),
                                &inputs_);
}

uint64_t BenchmarkTfLiteModel::ComputeInputBytes() {
  TFLITE_BENCHMARK_CHECK(interpreter_);
  uint64_t total_input_bytes = 0;
  for (int input : interpreter_->inputs()) {
    auto* t = interpreter_->tensor(input);
    total_input_bytes += t->bytes;
  }
  return total_input_bytes;
}

TfLiteStatus BenchmarkTfLiteModel::PrepareInputData() {
  auto interpreter_inputs = interpreter_->inputs();
  const size_t input_size = interpreter_inputs.size();
  CleanUp();

  for (int j = 0; j < input_size; ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    std::vector<int> sizes = TfLiteIntArrayToVector(t->dims);
    int num_elements = 1;
    for (int i = 0; i < sizes.size(); ++i) {
      num_elements *= sizes[i];
    }
    InputTensorData t_data;
    if (t->type == kTfLiteFloat32) {
      t_data.bytes = sizeof(float) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<float>(t_data.data.f, num_elements, []() {
        return static_cast<float>(rand()) / RAND_MAX - 0.5f;
      });
    } else if (t->type == kTfLiteFloat16) {
      t_data.bytes = sizeof(TfLiteFloat16) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
#if __GNUC__ && \
    (__clang__ || __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE)
      // __fp16 is available on Clang or when __ARM_FP16_FORMAT_* is defined.
      FillRandomValue<TfLiteFloat16>(
          t_data.data.f16, num_elements, []() -> TfLiteFloat16 {
            __fp16 f16_value = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            TfLiteFloat16 f16_placeholder_value;
            memcpy(&f16_placeholder_value, &f16_value, sizeof(TfLiteFloat16));
            return f16_placeholder_value;
          });
#else
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t->name
                        << " of type FLOAT16 on this platform.";
#endif
    } else if (t->type == kTfLiteInt64) {
      t_data.bytes = sizeof(int64_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<int64_t>(t_data.data.i64, num_elements, []() {
        return static_cast<int64_t>(rand()) % 100;
      });
    } else if (t->type == kTfLiteInt32) {
      // TODO(yunluli): This is currently only used for handling embedding input
      // for speech models. Generalize if necessary.
      t_data.bytes = sizeof(int32_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<int32_t>(t_data.data.i32, num_elements, []() {
        return static_cast<int32_t>(rand()) % 100;
      });
    } else if (t->type == kTfLiteInt16) {
      t_data.bytes = sizeof(int16_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<int16_t>(t_data.data.i16, num_elements, []() {
        return static_cast<int16_t>(rand()) % 100;
      });
    } else if (t->type == kTfLiteUInt8) {
      t_data.bytes = sizeof(uint8_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<uint8_t>(t_data.data.uint8, num_elements, []() {
        return static_cast<uint8_t>(rand()) % 255;
      });
    } else if (t->type == kTfLiteInt8) {
      t_data.bytes = sizeof(int8_t) * num_elements;
      t_data.data.raw = new char[t_data.bytes];
      FillRandomValue<int8_t>(t_data.data.int8, num_elements, []() {
        return static_cast<int8_t>(rand()) % 255 - 127;
      });
    } else if (t->type == kTfLiteString) {
      // TODO(haoliang): No need to cache string tensors right now.
    } else {
      TFLITE_LOG(ERROR) << "Don't know how to populate tensor " << t->name
                        << " of type " << t->type;
      return kTfLiteError;
    }
    inputs_data_.push_back(t_data);
  }
  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::ResetInputsAndOutputs() {
  auto interpreter_inputs = interpreter_->inputs();
  // Set the values of the input tensors from inputs_data_.
  for (int j = 0; j < interpreter_inputs.size(); ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type == kTfLiteFloat32) {
      std::memcpy(interpreter_->typed_tensor<float>(i), inputs_data_[j].data.f,
                  inputs_data_[j].bytes);
    } else if (t->type == kTfLiteFloat16) {
      std::memcpy(interpreter_->typed_tensor<TfLiteFloat16>(i),
                  inputs_data_[j].data.f16, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteInt64) {
      std::memcpy(interpreter_->typed_tensor<int64_t>(i),
                  inputs_data_[j].data.i64, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteInt32) {
      std::memcpy(interpreter_->typed_tensor<int32_t>(i),
                  inputs_data_[j].data.i32, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteInt64) {
      std::memcpy(interpreter_->typed_tensor<int64_t>(i),
                  inputs_data_[j].data.i64, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteInt16) {
      std::memcpy(interpreter_->typed_tensor<int16_t>(i),
                  inputs_data_[j].data.i16, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteUInt8) {
      std::memcpy(interpreter_->typed_tensor<uint8_t>(i),
                  inputs_data_[j].data.uint8, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteInt8) {
      std::memcpy(interpreter_->typed_tensor<int8_t>(i),
                  inputs_data_[j].data.int8, inputs_data_[j].bytes);
    } else if (t->type == kTfLiteString) {
      tflite::DynamicBuffer buffer;
      std::vector<int> sizes = TfLiteIntArrayToVector(t->dims);
      FillRandomString(&buffer, sizes, []() {
        return "we're have some friends over saturday to hang out in the yard";
      });
      buffer.WriteToTensor(interpreter_->tensor(i), /*new_shape=*/nullptr);
    } else {
      TFLITE_LOG(ERROR) << "Don't know how to populate tensor " << t->name
                        << " of type " << t->type;
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::Init() {
  std::string graph = params_.Get<std::string>("graph");
  model_ = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model_) {
    TFLITE_LOG(ERROR) << "Failed to mmap model " << graph;
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Loaded model " << graph;
  model_->error_reporter();
  TFLITE_LOG(INFO) << "resolved reporter";

  auto resolver = GetOpResolver();

  const int32_t num_threads = params_.Get<int32_t>("num_threads");
  tflite::InterpreterBuilder(*model_, *resolver)(&interpreter_, num_threads);
  if (!interpreter_) {
    TFLITE_LOG(ERROR) << "Failed to construct interpreter";
    return kTfLiteError;
  }

  interpreter_->UseNNAPI(params_.Get<bool>("use_legacy_nnapi"));

  delegates_ = GetDelegates();
  for (const auto& delegate : delegates_) {
    if (interpreter_->ModifyGraphWithDelegate(delegate.second.get()) !=
        kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to apply " << delegate.first << " delegate.";
      return kTfLiteError;
    } else {
      if (params_.Get<bool>("require_full_delegation")) {
        bool fully_delegated = true;
        if (interpreter_->execution_plan().size() != 1) {
          fully_delegated = false;
        } else {
          int first_node_id = interpreter_->execution_plan()[0];
          const TfLiteNode first_node =
              interpreter_->node_and_registration(first_node_id)->first;
          if (delegate.second.get() != first_node.delegate) {
            fully_delegated = false;
          }
        }

        if (!fully_delegated) {
          TFLITE_LOG(ERROR) << "Disallowed CPU fallback detected.";
          return kTfLiteError;
        }
      }

      TFLITE_LOG(INFO) << "Applied " << delegate.first << " delegate.";
    }
  }

  interpreter_->SetAllowFp16PrecisionForFp32(params_.Get<bool>("allow_fp16"));

  auto interpreter_inputs = interpreter_->inputs();

  if (!inputs_.empty()) {
    TFLITE_BENCHMARK_CHECK_EQ(inputs_.size(), interpreter_inputs.size())
        << "Inputs mismatch: Model inputs #:" << interpreter_inputs.size()
        << " expected: " << inputs_.size();
  }

  // Check if the tensor names match, and log a warning if it doesn't.
  // TODO(ycling): Consider to make this an error again when the new converter
  // create tensors with consistent naming.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (input.name != t->name) {
      TFLITE_LOG(WARN) << "Tensor # " << i << " is named " << t->name
                       << " but flags call it " << input.name;
    }
  }

  // Resize all non-string tensors.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type != kTfLiteString) {
      interpreter_->ResizeInputTensor(i, input.shape);
    }
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to allocate tensors!";
    return kTfLiteError;
  }

  // Install profilers if necessary.
  if (params_.Get<bool>("enable_op_profiling")) {
    profiling_listener_.reset(new ProfilingListener(
        interpreter_.get(),
        params_.Get<int32_t>("max_profiling_buffer_entries")));
    AddListener(profiling_listener_.get());
  }
#ifdef GEMMLOWP_PROFILING
  gemmlowp_profiling_listener_.reset(new GemmlowpProfilingListener());
  AddListener(gemmlowp_profiling_listener_.get());
#endif

  return kTfLiteOk;
}

#if defined(__ANDROID__)
bool IsValidGLObjectTypeInGPU(int32_t type) {
  if (type < TFLITE_GL_OBJECT_TYPE_FASTEST ||
      type > TFLITE_GL_OBJECT_TYPE_BUFFER) {
    TFLITE_LOG(WARN) << "The specified GL object type in GPU is invalid: "
                     << type;
    return false;
  }
  return true;
}
#endif

BenchmarkTfLiteModel::TfLiteDelegatePtrMap BenchmarkTfLiteModel::GetDelegates()
    const {
  TfLiteDelegatePtrMap delegates;
  if (params_.Get<bool>("use_gpu")) {
#if defined(__ANDROID__)
    TfLiteGpuDelegateOptions gpu_opts = TfLiteGpuDelegateOptionsDefault();
    gpu_opts.metadata =
        model_ ? TfLiteGpuDelegateGetModelMetadata(model_->GetModel())
               : nullptr;
    gpu_opts.compile_options.precision_loss_allowed =
        params_.Get<bool>("gpu_precision_loss_allowed") ? 1 : 0;
    int32_t gl_obj_type = params_.Get<int32_t>("gpu_gl_object_type");
    // We overwrite the gl object type to the recommended value if the specified
    // isn't valid.
    if (!IsValidGLObjectTypeInGPU(gl_obj_type)) {
      gl_obj_type = TFLITE_GL_OBJECT_TYPE_FASTEST;
    }
    gpu_opts.compile_options.preferred_gl_object_type = gl_obj_type;
    gpu_opts.compile_options.dynamic_batch_enabled = 0;
    Interpreter::TfLiteDelegatePtr delegate =
        evaluation::CreateGPUDelegate(model_.get(), &gpu_opts);
#else
    TFLITE_LOG(WARN) << "The GPU delegate compile options aren't supported to "
                        "be benchmarked on non-Android platforms.";
    Interpreter::TfLiteDelegatePtr delegate =
        evaluation::CreateGPUDelegate(model_.get());
#endif

    if (!delegate) {
      TFLITE_LOG(WARN) << "GPU acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("GPU", std::move(delegate));
    }
  }
  if (params_.Get<bool>("use_nnapi")) {
    StatefulNnApiDelegate::Options options;
    std::string accelerator_name =
        params_.Get<std::string>("nnapi_accelerator_name");
    if (!accelerator_name.empty()) {
      options.accelerator_name = accelerator_name.c_str();
    }
    Interpreter::TfLiteDelegatePtr delegate =
        evaluation::CreateNNAPIDelegate(options);
    if (!delegate) {
      TFLITE_LOG(WARN) << "NNAPI acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("NNAPI", std::move(delegate));
    }
  } else if (!params_.Get<std::string>("nnapi_accelerator_name").empty()) {
    TFLITE_LOG(WARN)
        << "`--use_nnapi=true` must be set for the provided NNAPI accelerator ("
        << params_.Get<std::string>("nnapi_accelerator_name")
        << ") to be used.";
  }
  return delegates;
}

std::unique_ptr<tflite::OpResolver> BenchmarkTfLiteModel::GetOpResolver()
    const {
  tflite::OpResolver* resolver = nullptr;
#ifdef TFLITE_CUSTOM_OPS_HEADER
  resolver = new tflite::MutableOpResolver();
  RegisterSelectedOps(static_cast<tflite::MutableOpResolver*>(resolver));
#else
  resolver = new tflite::ops::builtin::BuiltinOpResolver();
#endif

  return std::unique_ptr<tflite::OpResolver>(resolver);
}

TfLiteStatus BenchmarkTfLiteModel::RunImpl() { return interpreter_->Invoke(); }

}  // namespace benchmark
}  // namespace tflite

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"

namespace tensorflow {
namespace data {

string ToString(CompressionType compression_type) {
  switch (compression_type) {
    case CompressionType::ZLIB:
      return "ZLIB";
    case CompressionType::GZIP:
      return "GZIP";
    case CompressionType::RAW:
      return "RAW";
    case CompressionType::UNCOMPRESSED:
      return "";
  }
}

io::ZlibCompressionOptions GetZlibCompressionOptions(
    CompressionType compression_type) {
  switch (compression_type) {
    case CompressionType::ZLIB:
      return io::ZlibCompressionOptions::DEFAULT();
    case CompressionType::GZIP:
      return io::ZlibCompressionOptions::GZIP();
    case CompressionType::RAW:
      return io::ZlibCompressionOptions::RAW();
    case CompressionType::UNCOMPRESSED:
      LOG(WARNING) << "ZlibCompressionOptions does not have an option for "
                   << ToString(compression_type);
      return io::ZlibCompressionOptions::DEFAULT();
  }
}

Status WriteDataToFile(const string& filename, const char* data) {
  return WriteDataToFile(filename, data, CompressionParams());
}

Status WriteDataToFile(const string& filename, const char* data,
                       const CompressionParams& params) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file_writer;
  TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file_writer));
  if (params.compression_type == CompressionType::UNCOMPRESSED) {
    TF_RETURN_IF_ERROR(file_writer->Append(data));
  } else if (params.compression_type == CompressionType::ZLIB ||
             params.compression_type == CompressionType::GZIP ||
             params.compression_type == CompressionType::RAW) {
    auto zlib_compression_options =
        GetZlibCompressionOptions(params.compression_type);
    io::ZlibOutputBuffer out(file_writer.get(), params.input_buffer_size,
                             params.output_buffer_size,
                             zlib_compression_options);
    TF_RETURN_IF_ERROR(out.Init());
    TF_RETURN_IF_ERROR(out.Append(data));
    TF_RETURN_IF_ERROR(out.Flush());
    TF_RETURN_IF_ERROR(out.Close());
  } else {
    return tensorflow::errors::InvalidArgument(
        "Unsupported compression_type: ", ToString(params.compression_type));
  }

  TF_RETURN_IF_ERROR(file_writer->Flush());
  TF_RETURN_IF_ERROR(file_writer->Close());

  return Status::OK();
}

Status WriteDataToTFRecordFile(const string& filename,
                               const std::vector<absl::string_view>& records,
                               const CompressionParams& params) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file_writer;
  TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file_writer));
  auto options = io::RecordWriterOptions::CreateRecordWriterOptions(
      ToString(params.compression_type));
  options.zlib_options.input_buffer_size = params.input_buffer_size;
  io::RecordWriter record_writer(file_writer.get(), options);
  for (const auto& record : records) {
    TF_RETURN_IF_ERROR(record_writer.WriteRecord(record));
  }
  TF_RETURN_IF_ERROR(record_writer.Flush());
  TF_RETURN_IF_ERROR(record_writer.Close());
  TF_RETURN_IF_ERROR(file_writer->Flush());
  TF_RETURN_IF_ERROR(file_writer->Close());
  return Status::OK();
}

template <typename T>
Status IsEqual(const Tensor& t1, const Tensor& t2) {
  if (t1.dtype() != t2.dtype()) {
    return tensorflow::errors::Internal(
        "Two tensors have different dtypes: ", DataTypeString(t1.dtype()),
        " vs. ", DataTypeString(t2.dtype()));
  }
  if (!t1.IsSameSize(t2)) {
    return tensorflow::errors::Internal(
        "Two tensors have different shapes: ", t1.shape().DebugString(),
        " vs. ", t2.shape().DebugString());
  }

  auto flat_t1 = t1.flat<T>();
  auto flat_t2 = t2.flat<T>();
  auto length = flat_t1.size();

  for (int i = 0; i < length; ++i) {
    if (flat_t1(i) != flat_t2(i)) {
      return tensorflow::errors::Internal(
          "Two tensors have different values "
          "at [",
          i, "]: ", flat_t1(i), " vs. ", flat_t2(i));
    }
  }
  return Status::OK();
}

Status DatasetOpsTestBase::ExpectEqual(const Tensor& a, const Tensor& b) {
  switch (a.dtype()) {
#define CASE(DT)                           \
  case DataTypeToEnum<DT>::value:          \
    TF_RETURN_IF_ERROR(IsEqual<DT>(a, b)); \
    break;
    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_tstring(CASE);
    TF_CALL_uint32(CASE);
    TF_CALL_uint64(CASE);
    // TODO(feihugis): figure out how to support variant tensors.
#undef CASE
    default:
      return errors::Internal("Unsupported dtype: ", a.dtype());
  }
  return Status::OK();
}

template <typename T>
bool compare(const Tensor& t1, const Tensor& t2) {
  auto flat_t1 = t1.flat<T>();
  auto flat_t2 = t2.flat<T>();
  auto length = std::min(flat_t1.size(), flat_t2.size());
  for (int i = 0; i < length; ++i) {
    if (flat_t1(i) < flat_t2(i)) return true;
    if (flat_t1(i) > flat_t2(i)) return false;
  }
  return flat_t1.size() < length;
}

Status DatasetOpsTestBase::ExpectEqual(std::vector<Tensor> produced_tensors,
                                       std::vector<Tensor> expected_tensors,
                                       bool compare_order) {
  if (produced_tensors.size() != expected_tensors.size()) {
    return Status(tensorflow::errors::Internal(
        "The two tensor vectors have different size (", produced_tensors.size(),
        " v.s. ", expected_tensors.size(), ")"));
  }

  if (produced_tensors.empty()) return Status::OK();
  if (produced_tensors[0].dtype() != expected_tensors[0].dtype()) {
    return Status(tensorflow::errors::Internal(
        "The two tensor vectors have different dtypes (",
        produced_tensors[0].dtype(), " v.s. ", expected_tensors[0].dtype(),
        ")"));
  }

  if (!compare_order) {
    const DataType& dtype = produced_tensors[0].dtype();
    switch (dtype) {
#define CASE(DT)                                                \
  case DT:                                                      \
    std::sort(produced_tensors.begin(), produced_tensors.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    std::sort(expected_tensors.begin(), expected_tensors.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    break;
      CASE(DT_FLOAT);
      CASE(DT_DOUBLE);
      CASE(DT_INT32);
      CASE(DT_UINT8);
      CASE(DT_INT16);
      CASE(DT_INT8);
      CASE(DT_STRING);
      CASE(DT_INT64);
      CASE(DT_BOOL);
      CASE(DT_QINT8);
      CASE(DT_QUINT8);
      CASE(DT_QINT32);
      CASE(DT_QINT16);
      CASE(DT_QUINT16);
      CASE(DT_UINT16);
      CASE(DT_HALF);
      CASE(DT_UINT32);
      CASE(DT_UINT64);
      // TODO(feihugis): support other dtypes.
#undef CASE
      default:
        return errors::Internal("Unsupported dtype: ", dtype);
    }
  }

  for (int i = 0; i < produced_tensors.size(); ++i) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::ExpectEqual(produced_tensors[i],
                                                       expected_tensors[i]));
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateTensorSliceDatasetKernel(
    StringPiece node_name, const DataTypeVector& dtypes,
    const std::vector<PartialTensorShape>& shapes,
    std::unique_ptr<OpKernel>* tensor_slice_dataset_kernel) {
  std::vector<string> components;
  components.reserve(dtypes.size());
  for (int i = 0; i < dtypes.size(); ++i) {
    // Create the placeholder names for the input components of
    // `TensorSliceDataset`.
    components.emplace_back(strings::StrCat("component_", i));
  }
  NodeDef node_def = test::function::NDef(
      node_name, "TensorSliceDataset", components,
      {{"Toutput_types", dtypes}, {"output_shapes", shapes}});
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, tensor_slice_dataset_kernel));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateTensorSliceDataset(
    StringPiece node_name, std::vector<Tensor>* const components,
    DatasetBase** tensor_slice_dataset) {
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  DataTypeVector dtypes;
  dtypes.reserve(components->size());
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(components->size());
  for (const auto& t : *components) {
    dtypes.push_back(t.dtype());
    gtl::InlinedVector<int64, 4> partial_dim_sizes;
    for (int i = 1; i < t.dims(); ++i) {
      partial_dim_sizes.push_back(t.dim_size(i));
    }
    shapes.emplace_back(std::move(partial_dim_sizes));
  }
  TF_RETURN_IF_ERROR(CreateTensorSliceDatasetKernel(
      node_name, dtypes, shapes, &tensor_slice_dataset_kernel));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto& tensor : *components) {
    inputs.emplace_back(&tensor);
  }
  TF_RETURN_IF_ERROR(CheckOpKernelInput(*tensor_slice_dataset_kernel, inputs));
  std::unique_ptr<OpKernelContext> context;
  TF_RETURN_IF_ERROR(CreateOpKernelContext(tensor_slice_dataset_kernel.get(),
                                           &inputs, &context));
  TF_RETURN_IF_ERROR(
      RunOpKernel(tensor_slice_dataset_kernel.get(), context.get()));
  TF_RETURN_IF_ERROR(
      GetDatasetFromContext(context.get(), 0, tensor_slice_dataset));
  return Status::OK();
}

// Create a `RangeDataset` dataset as a variant tensor.
Status DatasetOpsTestBase::MakeRangeDataset(
    const Tensor& start, const Tensor& stop, const Tensor& step,
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes,
    Tensor* range_dataset) {
  GraphConstructorOptions graph_opts;
  graph_opts.allow_internal_ops = true;
  graph_opts.expect_device_spec = false;
  TF_RETURN_IF_ERROR(
      RunFunction(test::function::MakeRangeDataset(),
                  /*attrs*/
                  {{RangeDatasetOp::kOutputTypes, output_types},
                   {RangeDatasetOp::kOutputShapes, output_shapes}},
                  /*inputs*/ {start, stop, step}, graph_opts,
                  /*rets*/ {range_dataset}));
  return Status::OK();
}

// Create a `RangeDataset` dataset as a variant tensor.
Status DatasetOpsTestBase::MakeRangeDataset(
    const RangeDatasetParams& range_dataset_params, Tensor* range_dataset) {
  GraphConstructorOptions graph_opts;
  graph_opts.allow_internal_ops = true;
  graph_opts.expect_device_spec = false;
  TF_RETURN_IF_ERROR(RunFunction(
      test::function::MakeRangeDataset(),
      /*attrs*/
      {{RangeDatasetOp::kOutputTypes, range_dataset_params.output_dtypes},
       {RangeDatasetOp::kOutputShapes, range_dataset_params.output_shapes}},
      /*inputs*/
      {range_dataset_params.start, range_dataset_params.stop,
       range_dataset_params.step},
      graph_opts,
      /*rets*/ {range_dataset}));
  return Status::OK();
}

// Create a `TakeDataset` dataset as a variant tensor.
Status DatasetOpsTestBase::MakeTakeDataset(
    const Tensor& input_dataset, int64 count,
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes,
    Tensor* take_dataset) {
  GraphConstructorOptions graph_opts;
  graph_opts.allow_internal_ops = true;
  graph_opts.expect_device_spec = false;

  Tensor count_tensor = CreateTensor<int64>(TensorShape({}), {count});
  TF_RETURN_IF_ERROR(
      RunFunction(test::function::MakeTakeDataset(),
                  /*attrs*/
                  {{TakeDatasetOp::kOutputTypes, output_types},
                   {TakeDatasetOp::kOutputShapes, output_shapes}},
                  /*inputs*/ {input_dataset, count_tensor}, graph_opts,
                  /*rets*/ {take_dataset}));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernel(
    const NodeDef& node_def, std::unique_ptr<OpKernel>* op_kernel) {
  OpKernel* kernel;
  TF_RETURN_IF_ERROR(tensorflow::CreateOpKernel(device_type_, device_.get(),
                                                allocator_, flr_, node_def,
                                                TF_GRAPH_DEF_VERSION, &kernel));
  op_kernel->reset(kernel);
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDatasetContext(
    OpKernel* const dateset_kernel,
    gtl::InlinedVector<TensorValue, 4>* const inputs,
    std::unique_ptr<OpKernelContext>* dataset_context) {
  TF_RETURN_IF_ERROR(CheckOpKernelInput(*dateset_kernel, *inputs));
  TF_RETURN_IF_ERROR(
      CreateOpKernelContext(dateset_kernel, inputs, dataset_context));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDataset(OpKernel* kernel,
                                         OpKernelContext* context,
                                         DatasetBase** const dataset) {
  TF_RETURN_IF_ERROR(RunOpKernel(kernel, context));
  // Assume that DatasetOp has only one output.
  DCHECK_EQ(context->num_outputs(), 1);
  TF_RETURN_IF_ERROR(GetDatasetFromContext(context, 0, dataset));
  return Status::OK();
}

Status DatasetOpsTestBase::RestoreIterator(
    IteratorContext* ctx, IteratorStateReader* reader,
    const string& output_prefix, const DatasetBase& dataset,
    std::unique_ptr<IteratorBase>* iterator) {
  TF_RETURN_IF_ERROR(dataset.MakeIterator(ctx, output_prefix, iterator));
  TF_RETURN_IF_ERROR((*iterator)->Restore(ctx, reader));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateIteratorContext(
    OpKernelContext* const op_context,
    std::unique_ptr<IteratorContext>* iterator_context) {
  IteratorContext::Params params(op_context);
  params.resource_mgr = op_context->resource_manager();
  function_handle_cache_ = absl::make_unique<FunctionHandleCache>(flr_);
  params.function_handle_cache = function_handle_cache_.get();
  params.cancellation_manager = cancellation_manager_.get();
  *iterator_context = absl::make_unique<IteratorContext>(params);
  return Status::OK();
}

Status DatasetOpsTestBase::GetDatasetFromContext(OpKernelContext* context,
                                                 int output_index,
                                                 DatasetBase** const dataset) {
  Tensor* output = context->mutable_output(output_index);
  Status status = GetDatasetFromVariantTensor(*output, dataset);
  (*dataset)->Ref();
  return status;
}

Status DatasetOpsTestBase::InitThreadPool(int thread_num) {
  if (thread_num < 1) {
    return errors::InvalidArgument(
        "The `thread_num` argument should be positive but got: ", thread_num);
  }
  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      Env::Default(), ThreadOptions(), "test_thread_pool", thread_num);
  return Status::OK();
}

Status DatasetOpsTestBase::InitFunctionLibraryRuntime(
    const std::vector<FunctionDef>& flib, int cpu_num) {
  if (cpu_num < 1) {
    return errors::InvalidArgument(
        "The `cpu_num` argument should be positive but got: ", cpu_num);
  }
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", cpu_num});
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));
  resource_mgr_ = absl::make_unique<ResourceMgr>("default_container");

  FunctionDefLibrary proto;
  for (const auto& fdef : flib) *(proto.add_function()) = fdef;
  lib_def_ =
      absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  OptimizerOptions opts;
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
      opts, thread_pool_.get(), nullptr /* cluster_flr */);
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  if (thread_pool_ == nullptr) {
    runner_ = [](std::function<void()> fn) { fn(); };
  } else {
    runner_ = [this](std::function<void()> fn) {
      thread_pool_->Schedule(std::move(fn));
    };
  }
  return Status::OK();
}

Status DatasetOpsTestBase::RunOpKernel(OpKernel* op_kernel,
                                       OpKernelContext* context) {
  device_->Compute(op_kernel, context);
  return context->status();
}

Status DatasetOpsTestBase::RunFunction(
    const FunctionDef& fdef, test::function::Attrs attrs,
    const std::vector<Tensor>& args,
    const GraphConstructorOptions& graph_options, std::vector<Tensor*> rets) {
  std::unique_ptr<Executor> exec;
  InstantiationResult result;
  auto GetOpSig = [](const string& op, const OpDef** sig) {
    return OpRegistry::Global()->LookUpOpDef(op, sig);
  };
  TF_RETURN_IF_ERROR(InstantiateFunction(fdef, attrs, GetOpSig, &result));

  DataTypeVector arg_types = result.arg_types;
  DataTypeVector ret_types = result.ret_types;

  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_RETURN_IF_ERROR(
      ConvertNodeDefsToGraph(graph_options, result.nodes, g.get()));

  const int version = g->versions().producer();
  LocalExecutorParams params;
  params.function_library = flr_;
  params.device = device_.get();
  params.create_kernel = [this, version](const NodeDef& ndef,
                                         OpKernel** kernel) {
    return CreateNonCachedKernel(device_.get(), this->flr_, ndef, version,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };
  params.rendezvous_factory = [](const int64, const DeviceMgr* device_mgr,
                                 Rendezvous** r) {
    *r = new IntraProcessRendezvous(device_mgr);
    return Status::OK();
  };

  Executor* cur_exec;
  TF_RETURN_IF_ERROR(NewLocalExecutor(params, std::move(g), &cur_exec));
  exec.reset(cur_exec);
  FunctionCallFrame frame(arg_types, ret_types);
  TF_RETURN_IF_ERROR(frame.SetArgs(args));
  Executor::Args exec_args;
  exec_args.call_frame = &frame;
  exec_args.runner = runner_;
  TF_RETURN_IF_ERROR(exec->Run(exec_args));
  std::vector<Tensor> computed;
  TF_RETURN_IF_ERROR(frame.GetRetvals(&computed));
  if (computed.size() != rets.size()) {
    return errors::InvalidArgument(
        "The result does not match the expected number of return outpus",
        ". Expected: ", rets.size(), ". Actual: ", computed.size());
  }
  for (int i = 0; i < rets.size(); ++i) {
    *(rets[i]) = computed[i];
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
    std::unique_ptr<OpKernelContext>* context) {
  params_ = absl::make_unique<OpKernelContext::Params>();
  cancellation_manager_ = absl::make_unique<CancellationManager>();
  params_->cancellation_manager = cancellation_manager_.get();
  params_->device = device_.get();
  params_->frame_iter = FrameAndIter(0, 0);
  params_->function_library = flr_;
  params_->inputs = inputs;
  params_->op_kernel = kernel;
  params_->resource_manager = resource_mgr_.get();
  params_->runner = &runner_;
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
  slice_reader_cache_ =
      absl::make_unique<checkpoint::TensorSliceReaderCacheWrapper>();
  params_->slice_reader_cache = slice_reader_cache_.get();
  step_container_ =
      absl::make_unique<ScopedStepContainer>(0, [](const string&) {});
  params_->step_container = step_container_.get();

  // Set the allocator attributes for the outputs.
  allocator_attrs_.clear();
  for (int index = 0; index < params_->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params_->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    allocator_attrs_.emplace_back(attr);
  }
  params_->output_attr_array = gtl::vector_as_array(&allocator_attrs_);

  *context = absl::make_unique<OpKernelContext>(params_.get());
  return Status::OK();
}

Status DatasetOpsTestBase::CreateSerializationContext(
    std::unique_ptr<SerializationContext>* context) {
  *context =
      absl::make_unique<SerializationContext>(SerializationContext::Params{});
  return Status::OK();
}

Status DatasetOpsTestBase::CheckOpKernelInput(
    const OpKernel& kernel, const gtl::InlinedVector<TensorValue, 4>& inputs) {
  if (kernel.input_types().size() != inputs.size()) {
    return errors::Internal("The number of input elements should be ",
                            kernel.input_types().size(),
                            ", but got: ", inputs.size());
  }
  return Status::OK();
}

Status DatasetOpsTestBase::AddDatasetInput(
    gtl::InlinedVector<TensorValue, 4>* inputs, DataTypeVector input_types,
    DataType dtype, const TensorShape& shape) {
  if (input_types.size() < inputs->size()) {
    return errors::InvalidArgument("Adding more inputs than types: ",
                                   inputs->size(), " vs. ", input_types.size());
  }
  bool is_ref = IsRefType(input_types[inputs->size()]);
  std::unique_ptr<Tensor> input =
      absl::make_unique<Tensor>(allocator_, dtype, shape);

  if (is_ref) {
    DataType expected_dtype = RemoveRefType(input_types[inputs->size()]);
    if (expected_dtype != dtype) {
      return errors::InvalidArgument("The input data type is ", dtype,
                                     " , but expected: ", expected_dtype);
    }
    inputs->push_back({&lock_for_refs_, input.get()});
  } else {
    if (input_types[inputs->size()] != dtype) {
      return errors::InvalidArgument(
          "The input data type is ", dtype,
          " , but expected: ", input_types[inputs->size()]);
    }
    inputs->push_back({nullptr, input.get()});
  }

  // TODO(jsimsa): Figure out how to avoid using a member variable to garbage
  // collect the inputs.
  tensors_.push_back(std::move(input));

  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorGetNext(
    const std::vector<Tensor>& expected_outputs, bool compare_order) {
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_RETURN_IF_ERROR(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order=*/compare_order));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetNodeName(
    const string& expected_dataset_node_name) {
  EXPECT_EQ(dataset_->node_name(), expected_dataset_node_name);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetTypeString(
    const string& expected_type_str) {
  EXPECT_EQ(dataset_->type_string(), expected_type_str);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
  TF_EXPECT_OK(
      VerifyTypesMatch(dataset_->output_dtypes(), expected_output_dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(dataset_->output_shapes(),
                                      expected_output_shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetCardinality(int expected_cardinality) {
  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
  TF_EXPECT_OK(
      VerifyTypesMatch(iterator_->output_dtypes(), expected_output_dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(iterator_->output_shapes(),
                                      expected_output_shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorPrefix(
    const string& expected_iterator_prefix) {
  EXPECT_EQ(iterator_->prefix(), expected_iterator_prefix);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorSaveAndRestore(
    const string& iterator_prefix, const std::vector<Tensor>& expected_outputs,
    const std::vector<int>& breakpoints) {
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(
      dataset_->MakeIterator(iterator_ctx_.get(), iterator_prefix, &iterator));
  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_RETURN_IF_ERROR(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  auto expected_outputs_it = expected_outputs.begin();
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_RETURN_IF_ERROR(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader, iterator_prefix,
                                 *dataset_, &iterator));

    while (cur_iteration <= breakpoint) {
      TF_RETURN_IF_ERROR(iterator->GetNext(iterator_ctx_.get(), &out_tensors,
                                           &end_of_sequence));
      if (!end_of_sequence) {
        EXPECT_NE(expected_outputs_it, expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(out_tensors.back(), *expected_outputs_it));
        expected_outputs_it++;
      }
      cur_iteration++;
    }

    if (breakpoint >= expected_outputs.size()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow

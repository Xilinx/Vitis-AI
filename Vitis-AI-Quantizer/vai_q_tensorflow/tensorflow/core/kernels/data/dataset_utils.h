/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

template <typename T>
class AnonymousResourceOp : public OpKernel {
 public:
  static std::atomic<int64> resource_id_counter_;

  explicit AnonymousResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    FunctionLibraryRuntime* lib;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    OP_REQUIRES_OK(
        ctx, ctx->function_library()->Clone(&flib_def, &pflr, &lib, true));
    T* resource;
    OP_REQUIRES_OK(ctx, CreateResource(ctx, std::move(flib_def),
                                       std::move(pflr), lib, &resource));

    string container_name = name();
    string unique_name =
        strings::StrCat(container_name, resource_id_counter_.fetch_add(1));
    ResourceMgr* mgr = ctx->resource_manager();
    OP_REQUIRES_OK(ctx, mgr->Create<T>(container_name, unique_name, resource));

    Tensor* handle_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle_t));
    ResourceHandle handle = MakeResourceHandle(ctx, container_name, unique_name,
                                               MakeTypeIndex<T>());
    handle_t->scalar<ResourceHandle>()() = handle;

    if (create_deleter_) {
      Tensor* deleter_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &deleter_t));
      deleter_t->scalar<Variant>()() =
          ResourceDeleter(handle, ctx->resource_manager());
    }
  }

 protected:
  virtual string name() = 0;

  virtual Status CreateResource(
      OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* lib, T** resource) = 0;

  bool create_deleter_ = true;
};

template <typename T>
std::atomic<int64> AnonymousResourceOp<T>::resource_id_counter_;

// Returns a GraphDef representation of the given dataset.
Status AsGraphDef(OpKernelContext* ctx, const DatasetBase* dataset,
                  SerializationContext&& serialization_ctx,
                  GraphDef* graph_def);

// Creates a connection between "child" and "parent" cancellation managers so
// that parent cancellations are propagated to the child, returning a function
// that can be used to remove the connection.
Status ConnectCancellationManagers(CancellationManager* parent,
                                   CancellationManager* child,
                                   std::function<void()>* deregister_fn);

// Rewrites the input dataset using the given config.
Status RewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                      std::function<RewriterConfig(void)> config_factory,
                      bool optimize_function_library,
                      DatasetBase** rewritten_input);

// Returns Status::OK() if `expected` and `received` types match,
// errors::InvalidArgument otherwise.
Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received);

// Returns Status::OK() if `expected` and `received` shapes are compatible,
// errors::InvalidArgument otherwise.
Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received);

// Returns a stable hash of the portion of the graph `g` rooted at
// `node`, by creating a Merkle tree-like structure.
//
// Specifically, this function recursively walks the graph from `node` by
// following its inputs.
//
// The hash is computed by hashing its op name, device, attributes, and hashes
// of its inputs (if applicable).
//
// There is currently no guarantee that the hash of a subgraph will stay the
// same between TensorFlow builds.
uint64 HashSubgraph(const GraphDef& g, const NodeDef* node);

// Returns a stable hash of the function `f`.
//
// This function computes the hash by hashing the metadata of the
// function (disregarding the auto-generated names and descriptions) and also
// hashing the subgraph rooted at each of the output nodes.
//
// There is currently no guarantee that the hash of a function will stay the
// same between TensorFlow builds.
uint64 HashSubgraphFunction(const FunctionDefLibrary& library,
                            const FunctionDef* f);

// Helper class for reading data from a VariantTensorData object.
class VariantTensorDataReader : public IteratorStateReader {
 public:
  explicit VariantTensorDataReader(const VariantTensorData* data);

  // Returns OK iff the initialization was successful.
  Status ReadScalar(StringPiece key, int64* val) override;
  Status ReadScalar(StringPiece key, tstring* val) override;
  Status ReadTensor(StringPiece key, Tensor* val) override;
  bool Contains(StringPiece key) override;

 private:
  template <typename T>
  Status ReadScalarInternal(StringPiece key, T* val);
  Status ReadTensorInternal(StringPiece key, Tensor* val);

  std::map<string, size_t> map_;
  const VariantTensorData* data_;  // Not owned.
};

// Helper class for writing data to a VariantTensorData object.
class VariantTensorDataWriter : public IteratorStateWriter {
 public:
  // Does not take ownership of data.
  explicit VariantTensorDataWriter(VariantTensorData* data) : data_(data) {}
  Status WriteScalar(StringPiece key, const int64 val) override;
  Status WriteScalar(StringPiece key, const tstring& val) override;
  Status WriteTensor(StringPiece key, const Tensor& val) override;

  // Writes the metadata to `data_`.
  Status Flush();

 private:
  template <typename T>
  Status WriteScalarInternal(StringPiece key, const T& val);
  Status WriteTensorInternal(StringPiece key, const Tensor& val);

  VariantTensorData* data_;
  std::vector<string> keys_;
};

// Adds the functions in `to_add` to `base`. If a function with a matching
// signature already exists in `base`, replaces it with the function from
// `to_add`.
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionLibraryDefinition& to_add);
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionDefLibrary& to_add);

// Creates a runner that runs functions with limited parallelism.
std::function<void(std::function<void()>)> RunnerWithMaxParallelism(
    std::function<void(std::function<void()>)> runner, int max_parallelism);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_

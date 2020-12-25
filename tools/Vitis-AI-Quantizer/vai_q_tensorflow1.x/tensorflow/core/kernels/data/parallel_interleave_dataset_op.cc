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
#include "tensorflow/core/kernels/data/parallel_interleave_dataset_op.h"

#include <atomic>
#include <deque>
#include <memory>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kInputDataset;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOtherArguments;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kCycleLength;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kBlockLength;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const ParallelInterleaveDatasetOp::kFunc;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kTarguments;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOutputTypes;
/* static */ constexpr const char* const
    ParallelInterleaveDatasetOp::kOutputShapes;
/* static */ constexpr const char* const ParallelInterleaveDatasetOp::kSloppy;

constexpr char kDataParallelInterleaveWorkerPool[] =
    "data_parallel_interleave_worker_pool";
constexpr char kParallelism[] = "parallelism";
constexpr char kBlockIndex[] = "block_index";
constexpr char kCycleIndex[] = "cycle_index";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kElementIdCounter[] = "element_id_counter";
constexpr char kNumOpen[] = "num_open";
constexpr char kCurrentElements[] = "current_elements";
constexpr char kCurrentElementsSize[] = "current_elements.size";
constexpr char kFutureElements[] = "future_elements";
constexpr char kFutureElementsSize[] = "future_elements.size";
constexpr char kResultsSuffix[] = ".results";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessageSuffix[] = ".error_message";
constexpr char kIdSuffix[] = ".id";
constexpr char kSizeSuffix[] = ".size";
constexpr char kInputsSuffix[] = ".inputs";
constexpr char kIsReadySuffix[] = ".is_ready";
constexpr char kTFDataParallelInterleaveCurrent[] =
    "tf_data_parallel_interleave_current";
constexpr char kTFDataParallelInterleaveFuture[] =
    "tf_data_parallel_interleave_future";

// `kPrefetchFactor * cycle_length` is the number of future cycle elements that
// will be prefetched ahead of time. The purpose of prefetching future cycle
// elements is to overlap expensive initialization (e.g. opening of a remote
// file) with other computation.
constexpr double kPrefetchFactor = 2.0L;

// `kCPUFactor * port::NumSchedulableCPUs()` is the size of the threadpool
// created by this op. The rationale behind creating more threads than CPUs
// is to achieve efficient CPU utilization when some of the threads perform I/O.
constexpr double kCPUFactor = 2.0L;

// The motivation for creating an alternative implementation of parallel
// interleave is to decouple the degree of parallelism from the cycle length.
// This makes it possible to change the degree of parallelism (e.g. through
// auto-tuning) without changing the cycle length (which would change the order
// in which elements are produced).
//
// Furthermore, this class favors modularity over extended functionality. In
// particular, it refrains from implementing configurable buffering of output
// elements and prefetching of input iterators.
class ParallelInterleaveDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
          int64 block_length, int64 num_parallel_calls, bool sloppy,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        num_parallel_calls_(num_parallel_calls),
        sloppy_(sloppy),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    return absl::make_unique<ParallelInterleaveIterator>(
        ParallelInterleaveIterator::Params{
            this,
            name_utils::IteratorPrefix(
                ParallelInterleaveDatasetOp::kDatasetType, prefix, params)},
        sloppy_);
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    return name_utils::DatasetDebugString(
        ParallelInterleaveDatasetOp::kDatasetType, params);
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* cycle_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(cycle_length_, &cycle_length_node));
    Node* block_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(block_length_, &block_length_node));
    Node* num_parallel_calls_node;
    TF_RETURN_IF_ERROR(
        b->AddScalar(num_parallel_calls_, &num_parallel_calls_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
    AttrValue sloppy_attr;
    b->BuildAttrValue(sloppy_, &sloppy_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(this,
                                     {{0, input_node},
                                      {2, cycle_length_node},
                                      {3, block_length_node},
                                      {4, num_parallel_calls_node}},
                                     {{1, other_arguments}},
                                     {{kFunc, f},
                                      {kTarguments, other_arguments_types_attr},
                                      {kSloppy, sloppy_attr}},
                                     output));
    return Status::OK();
  }

 private:
  class ParallelInterleaveIterator : public DatasetIterator<Dataset> {
   public:
    ParallelInterleaveIterator(const Params& params, bool sloppy)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          cond_var_(std::make_shared<condition_variable>()),
          num_parallel_calls_(std::make_shared<model::SharedState>(
              params.dataset->num_parallel_calls_, mu_, cond_var_)),
          sloppy_(sloppy),
          current_elements_(params.dataset->cycle_length_) {}

    ~ParallelInterleaveIterator() override {
      mutex_lock l(*mu_);
      // Cancel the runner thread.
      cancelled_ = true;
      cond_var_->notify_all();
      // Wait for all in-flight calls to complete.
      while (current_num_calls_ > 0 || future_num_calls_ > 0) {
        cond_var_->wait(l);
      }
    }

    string BuildTraceMeName() override {
      // NOTE: We do not synchronize the following access to
      // num_parallel_calls_ to minimize the tracing overhead.
      int64 parallelism = num_parallel_calls_->value;
      return strings::StrCat(prefix(), "#parallelism=", parallelism, "#");
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);
      // The size of the threadpool `num_threads` is the smaller of:
      //
      // 1) The number of schedulable CPUs multiplied by a constant factor
      //    factor to account for the fact that some threads may perform I/O.
      //
      // 2) The maximum number of iterators instantiated at any given point
      //    in time (`cycle_length` for the current cycle elements and
      //    `kPrefetchFactor * cycle_length` for future cycle elements).
      //
      // Note that if `ctx->thread_pool()` is non-null, then instead of creating
      // a dedicated thread pool of size `num_threads`, computation will be
      // scheduled into the shared threadpool whose size is independent of
      // `num_threads`.
      const int num_threads = std::min(
          static_cast<int>(kCPUFactor * port::NumSchedulableCPUs()),
          static_cast<int>((kPrefetchFactor + 1) * dataset()->cycle_length_));
      thread_pool_ =
          ctx->CreateThreadPool(kDataParallelInterleaveWorkerPool, num_threads);
      if (num_parallel_calls_->value == model::kAutotune) {
        num_parallel_calls_->value = dataset()->cycle_length_;
      }
      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      std::shared_ptr<Result> result;
      {
        mutex_lock l(*mu_);
        EnsureThreadsStarted(ctx);
        while (!Consume(&result)) {
          RecordStop(ctx);
          cond_var_->wait(l);
          RecordStart(ctx);
        }
      }
      if (!result) {
        *end_of_sequence = true;
        return Status::OK();
      }
      if (result->status.ok()) {
        *out_tensors = std::move(result->return_values);
        RecordBufferDequeue(ctx, *out_tensors);
      }
      *end_of_sequence = false;
      return result->status;
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncInterleaveManyNode(
          std::move(args),
          {model::MakeParameter(kParallelism, num_parallel_calls_, /*min=*/1,
                                /*max=*/dataset()->cycle_length_)});
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(*mu_);
      // Wait for all in-flight calls to complete.
      while (current_num_calls_ > 0 || future_num_calls_ > 0) {
        cond_var_->wait(l);
      }
      DCHECK_EQ(current_num_calls_, 0);
      DCHECK_EQ(future_num_calls_, 0);
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kBlockIndex), block_index_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCycleIndex), cycle_index_));
      if (end_of_input_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kEndOfInput), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kElementIdCounter),
                                             element_id_counter_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kNumOpen), num_open_));
      TF_RETURN_IF_ERROR(WriteCurrentElements(writer));
      TF_RETURN_IF_ERROR(WriteFutureElements(writer));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kBlockIndex), &block_index_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCycleIndex), &cycle_index_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kElementIdCounter),
                                            &element_id_counter_));
      if (reader->Contains(full_name(kEndOfInput))) end_of_input_ = true;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumOpen), &num_open_));
      TF_RETURN_IF_ERROR(ReadCurrentElements(ctx, reader));
      TF_RETURN_IF_ERROR(ReadFutureElements(ctx, reader));
      return Status::OK();
    }

   private:
    // Represents the result of fetching an element from a dataset.
    struct Result {
      Status status;
      std::vector<Tensor> return_values;
      // Indicates whether the result is ready to be consumed.
      bool is_ready = false;
    };

    // The interleave transformation repeatedly inputs elements, applies the
    // user-provided function to transform the input elements to datasets, and
    // interleaves the elements of these datasets as its output.
    //
    // This structure represents an input element and derived state.
    struct Element {
      // Unique identifier, needed to support checkpointing.
      int64 id;
      // The actual input element.
      std::vector<Tensor> inputs;
      // Iterator created from the input element.
      std::unique_ptr<IteratorBase> iterator;
      mutex mu;
      // Buffer for storing the outputs of `iterator`.
      std::deque<std::shared_ptr<Result>> results GUARDED_BY(mu);
      // Indicates whether the element is used by a worker thread.
      bool in_use = false;
    };

    // Advances the position in the interleave cycle to the next cycle
    // element.
    void AdvanceToNextInCycle() EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      block_index_ = 0;
      cycle_index_ = (cycle_index_ + 1) % dataset()->cycle_length_;
    }

    // Advances the position in the interleave cycle by one.
    void AdvancePosition() EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      ++block_index_;
      if (block_index_ == dataset()->block_length_) {
        AdvanceToNextInCycle();
      }
    }

    // Consumes a result (if available), returning an indication of whether
    // a result is available. If `true` is returned, `result` either
    // points to a valid result or is null if end of input has been reached.
    bool Consume(std::shared_ptr<Result>* result)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!sloppy_) {
        return ConsumeHelper(result);
      }
      // If we are allowed to be sloppy (i.e. return results out of order),
      // try to find an element in the cycle that has a result available.
      for (int i = 0; i < dataset()->cycle_length_; ++i) {
        if (ConsumeHelper(result)) {
          return true;
        }
        AdvanceToNextInCycle();
      }
      return false;
    }

    bool ConsumeHelper(std::shared_ptr<Result>* result)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      while (true) {
        std::shared_ptr<Element> element = current_elements_[cycle_index_];
        if (element) {
          mutex_lock l(element->mu);
          if (!element->results.empty()) {
            if (element->results.front()->is_ready) {
              // We found a result.
              std::swap(*result, element->results.front());
              element->results.pop_front();
              AdvancePosition();
              cond_var_->notify_all();
              return true;
            } else {
              // Wait for the result to become ready.
              return false;
            }
          } else if (!element->iterator) {
            // We reached the end of input for this element. Reset
            // it and move on to the next cycle element.
            current_elements_[cycle_index_].reset();
            AdvanceToNextInCycle();
            cond_var_->notify_all();
            continue;
          } else {
            // Wait for the iterator to produce a result.
            return false;
          }
        } else {
          if (!future_elements_.empty() || !end_of_input_) {
            // Wait for an element to be created.
            return false;
          }
          // No new elements will be created; try to find a
          // non-empty element in the cycle.
          for (int i = 0; i < dataset()->cycle_length_; ++i) {
            AdvanceToNextInCycle();
            if (current_elements_[cycle_index_]) {
              break;
            }
          }
          if (current_elements_[cycle_index_]) {
            continue;
          }
          // End of input has been reached.
          return true;
        }
      }
    }

    // Manages current cycle elements, creating new iterators as needed and
    // asynchronously fetching results from existing iterators.
    //
    // This method runs in the `current_elements_manager_` background thread.
    void CurrentElementsManager(const std::shared_ptr<IteratorContext>& ctx) {
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
        const bool has_more_elements =
            !future_elements_.empty() || !end_of_input_;
        const int block_length = dataset()->block_length_;
        bool all_elements_busy = true;
        for (auto& element : current_elements_) {
          if (!element) {
            if (has_more_elements) {
              all_elements_busy = false;
              break;
            }
          } else {
            mutex_lock l(element->mu);
            if (!element->in_use && element->iterator &&
                element->results.size() < block_length) {
              all_elements_busy = false;
              break;
            }
          }
        }
        return all_elements_busy ||
               current_num_calls_ >= num_parallel_calls_->value;
      };
      while (true) {
        mutex_lock l(*mu_);

        // Wait until this thread is cancelled, the end of input has been
        // reached.
        while (!cancelled_ && (!end_of_input_ || num_open_ > 0) && busy()) {
          RecordStop(ctx.get());
          cond_var_->wait(l);
          RecordStart(ctx.get());
        }

        if (cancelled_ ||
            (future_elements_.empty() && end_of_input_ && num_open_ == 0)) {
          return;
        }

        for (int i = 0; i < dataset()->cycle_length_; ++i) {
          int idx = (cycle_index_ + i) % dataset()->cycle_length_;
          if (!current_elements_[idx]) {
            if (!future_elements_.empty()) {
              current_elements_[idx] = std::move(future_elements_.back());
              future_elements_.pop_back();
              if (current_elements_[idx]->iterator) {
                EnableAutotune(ctx.get(),
                               current_elements_[idx]->iterator.get());
              }
            } else {
              current_elements_[idx] = MakeElement(ctx);
              if (!current_elements_[idx]) {
                continue;
              }
            }
          }
          std::shared_ptr<Element> element = current_elements_[idx];
          if (!element->in_use && element->iterator) {
            int64 num_results;
            {
              mutex_lock l(element->mu);
              num_results = dataset()->block_length_ - element->results.size();
            }
            if (num_results > 0) {
              current_num_calls_++;
              element->in_use = true;
              thread_pool_->Schedule(std::bind(
                  &ParallelInterleaveIterator::FetchResults, this, ctx,
                  std::move(element), num_results,
                  [this, ctx]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
                    --current_num_calls_;
                    const auto& stats_aggregator = ctx->stats_aggregator();
                    if (stats_aggregator) {
                      stats_aggregator->AddScalar(
                          stats_utils::ThreadUtilizationScalarName(
                              dataset()->node_name()),
                          static_cast<float>(current_num_calls_) /
                              static_cast<float>(num_parallel_calls_->value),
                          num_elements());
                    }
                  }));
            }
          }
        }
        const auto& stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator) {
          stats_aggregator->AddScalar(
              stats_utils::ThreadUtilizationScalarName(dataset()->node_name()),
              static_cast<float>(current_num_calls_) /
                  static_cast<float>(num_parallel_calls_->value),
              num_elements());
        }
        cond_var_->notify_all();
      }
    }

    void EnsureThreadsStarted(IteratorContext* ctx)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!current_elements_manager_) {
        auto new_ctx = std::make_shared<IteratorContext>(*ctx);
        current_elements_manager_ = ctx->StartThread(
            kTFDataParallelInterleaveCurrent,
            [this, new_ctx]() { CurrentElementsManager(new_ctx); });
      }
      if (!future_elements_manager_) {
        auto new_ctx = std::make_shared<IteratorContext>(*ctx);
        future_elements_manager_ = ctx->StartThread(
            kTFDataParallelInterleaveFuture,
            [this, new_ctx]() { FutureElementsManager(new_ctx); });
      }
    }

    // Fetches up to `dataset()->block_length_` results from `element`.
    void FetchResults(const std::shared_ptr<IteratorContext>& ctx,
                      const std::shared_ptr<Element>& element,
                      int64 num_results, std::function<void()> done)
        LOCKS_EXCLUDED(*mu_) {
      RecordStart(ctx.get());
      bool end_of_input = false;
      for (int64 i = 0; i < num_results; ++i) {
        auto result = std::make_shared<Result>();
        result->status = element->iterator->GetNext(
            ctx.get(), &result->return_values, &end_of_input);
        if (end_of_input) {
          break;
        }
        RecordBufferEnqueue(ctx.get(), result->return_values);
        mutex_lock l(*mu_);
        mutex_lock l2(element->mu);
        element->results.push_back(result);
        result->is_ready = true;
        cond_var_->notify_all();
      }

      mutex_lock l(*mu_);
      // Release the ownership of the cycle element iterator.
      element->in_use = false;
      if (end_of_input) {
        // Close the iterator if end of input was encountered.
        element->iterator.reset();
        element->inputs.clear();
        --num_open_;
      }
      done();
      cond_var_->notify_all();
      RecordStop(ctx.get());
    }

    // Manages futures cycle elements, creating new iterators as needed and
    // asynchronously fetching results from existing iterators.
    //
    // This method runs in the `future_elements_manager_` background thread.
    void FutureElementsManager(const std::shared_ptr<IteratorContext>& ctx) {
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
        // TODO(jsimsa): Autotune the number of elements to prefetch.
        return future_elements_.size() >=
               static_cast<int>(kPrefetchFactor * dataset()->cycle_length_);
      };
      while (true) {
        mutex_lock l(*mu_);

        // Wait until this thread is cancelled, the end of input has been
        // reached, or the cycle element at the `cycle_index_` position is
        // not in use.
        while (!cancelled_ && !end_of_input_ && busy()) {
          RecordStop(ctx.get());
          cond_var_->wait(l);
          RecordStart(ctx.get());
        }

        if (cancelled_ || end_of_input_) {
          return;
        }

        while (!end_of_input_ && !busy()) {
          std::shared_ptr<Element> element = MakeElement(ctx);
          if (!element) {
            break;
          }
          future_elements_.push_front(element);
          if (!element->iterator) {
            continue;
          }
          DisableAutotune(ctx.get(), element->iterator.get());
          ++future_num_calls_;
          element->in_use = true;
          thread_pool_->Schedule(std::bind(
              &ParallelInterleaveIterator::FetchResults, this, ctx,
              std::move(element), dataset()->block_length_,
              [this]()
                  EXCLUSIVE_LOCKS_REQUIRED(*mu_) { --future_num_calls_; }));
        }
        cond_var_->notify_all();
      }
    }

    // Creates a new element.
    std::shared_ptr<Element> MakeElement(
        const std::shared_ptr<IteratorContext>& ctx)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      auto element = std::make_shared<Element>();
      element->id = element_id_counter_++;
      Status status =
          input_impl_->GetNext(ctx.get(), &element->inputs, &end_of_input_);
      if (!status.ok()) {
        auto result = std::make_shared<Result>();
        result->is_ready = true;
        result->status = status;
        mutex_lock l(element->mu);
        element->results.push_back(std::move(result));
        return element;
      }
      if (!end_of_input_) {
        Status status = MakeIteratorFromInputElement(
            ctx.get(), element->inputs, element->id,
            *instantiated_captured_func_, prefix(), &element->iterator);
        if (!status.ok()) {
          auto result = std::make_shared<Result>();
          result->is_ready = true;
          result->status = status;
          mutex_lock l(element->mu);
          element->results.push_back(std::move(result));
          return element;
        }
        ++num_open_;
      } else {
        element.reset();
      }
      return element;
    }

    Status WriteStatusLocked(IteratorStateWriter* writer,
                             const string& key_prefix, size_t idx,
                             const Status& status)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          CodeKey(key_prefix, idx), static_cast<int64>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(key_prefix, idx),
                                               status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatusLocked(IteratorStateReader* reader,
                            const string& key_prefix, size_t idx,
                            Status* status) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64 code_int;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(CodeKey(key_prefix, idx), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        tstring error_message;
        TF_RETURN_IF_ERROR(reader->ReadScalar(ErrorMessageKey(key_prefix, idx),
                                              &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    string CodeKey(const string& key_prefix, size_t idx) {
      return full_name(strings::StrCat(key_prefix, kResultsSuffix, "[", idx,
                                       "]", kCodeSuffix));
    }

    string ErrorMessageKey(const string& key_prefix, size_t idx) {
      return full_name(strings::StrCat(key_prefix, kResultsSuffix, "[", idx,
                                       "]", kErrorMessageSuffix));
    }

    Status WriteElement(std::shared_ptr<Element> element, int idx,
                        const string& key_prefix, IteratorStateWriter* writer)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (element->iterator) {
        TF_RETURN_IF_ERROR(SaveInput(writer, element->iterator));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kIdSuffix)),
            element->id));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kInputsSuffix,
                                      kSizeSuffix)),
            element->inputs.size()));
        for (int i = 0; i < element->inputs.size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(key_prefix, "[", idx, "]",
                                        kInputsSuffix, "[", i, "]")),
              element->inputs[i]));
        }
      }
      mutex_lock l(element->mu);
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                    kSizeSuffix)),
          element->results.size()));
      for (size_t i = 0; i < element->results.size(); i++) {
        std::shared_ptr<Result> result = element->results[i];
        TF_RETURN_IF_ERROR(WriteStatusLocked(
            writer, strings::StrCat(key_prefix, "[", idx, "]"), i,
            result->status));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                      "[", i, "]", kSizeSuffix)),
            result->return_values.size()));
        for (size_t j = 0; j < result->return_values.size(); j++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(key_prefix, "[", idx, "]",
                                        kResultsSuffix, "[", i, "][", j, "]")),
              result->return_values[j]));
        }
        if (result->is_ready) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(key_prefix, "[", idx, "]",
                                        kResultsSuffix, "[", i, "]",
                                        kIsReadySuffix)),
              ""));
        }
      }
      return Status::OK();
    }

    Status WriteCurrentElements(IteratorStateWriter* writer)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentElementsSize),
                                             current_elements_.size()));
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        if (current_elements_[idx]) {
          TF_RETURN_IF_ERROR(WriteElement(current_elements_[idx], idx,
                                          kCurrentElements, writer));
        }
      }
      return Status::OK();
    }

    Status WriteFutureElements(IteratorStateWriter* writer)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kFutureElementsSize),
                                             future_elements_.size()));
      for (int idx = 0; idx < future_elements_.size(); idx++) {
        if (future_elements_[idx]) {
          TF_RETURN_IF_ERROR(WriteElement(future_elements_[idx], idx,
                                          kFutureElements, writer));
        }
      }
      return Status::OK();
    }

    Status ReadElement(IteratorContext* ctx, IteratorStateReader* reader,
                       int idx, const string& key_prefix,
                       std::shared_ptr<Element>* out)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!reader->Contains(full_name(strings::StrCat(
              key_prefix, "[", idx, "]", kResultsSuffix, kSizeSuffix)))) {
        return Status::OK();
      }
      auto element = std::make_shared<Element>();
      mutex_lock l(element->mu);
      int64 results_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                    kSizeSuffix)),
          &results_size));
      element->results.resize(results_size);
      for (size_t i = 0; i < results_size; i++) {
        auto result = std::make_shared<Result>();
        TF_RETURN_IF_ERROR(
            ReadStatusLocked(reader, strings::StrCat(key_prefix, "[", idx, "]"),
                             i, &result->status));
        int64 num_return_values;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                      "[", i, "]", kSizeSuffix)),
            &num_return_values));
        result->return_values.reserve(num_return_values);
        for (size_t j = 0; j < num_return_values; j++) {
          result->return_values.emplace_back();
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              full_name(strings::StrCat(key_prefix, "[", idx, "]",
                                        kResultsSuffix, "[", i, "][", j, "]")),
              &result->return_values.back()));
        }
        result->is_ready = reader->Contains(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kResultsSuffix,
                                      "[", i, "]", kIsReadySuffix)));
        element->results[i] = std::move(result);
      }
      if (!reader->Contains(full_name(strings::StrCat(
              key_prefix, "[", idx, "]", kInputsSuffix, kSizeSuffix)))) {
        element->iterator.reset();
        *out = std::move(element);
        return Status::OK();
      }
      int64 inputs_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kInputsSuffix,
                                    kSizeSuffix)),
          &inputs_size));
      element->inputs.resize(inputs_size);
      for (int i = 0; i < inputs_size; i++) {
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            full_name(strings::StrCat(key_prefix, "[", idx, "]", kInputsSuffix,
                                      "[", i, "]")),
            &element->inputs[i]));
      }
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(key_prefix, "[", idx, "]", kIdSuffix)),
          &element->id));
      TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
          ctx, element->inputs, element->id, *instantiated_captured_func_.get(),
          prefix(), &element->iterator));
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, element->iterator));
      *out = std::move(element);
      return Status::OK();
    }

    Status ReadCurrentElements(IteratorContext* ctx,
                               IteratorStateReader* reader)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64 size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCurrentElementsSize), &size));
      DCHECK_EQ(current_elements_.size(), size);
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        TF_RETURN_IF_ERROR(ReadElement(ctx, reader, idx, kCurrentElements,
                                       &current_elements_[idx]));
      }
      return Status::OK();
    }

    Status ReadFutureElements(IteratorContext* ctx, IteratorStateReader* reader)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64 size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kFutureElementsSize), &size));
      future_elements_.resize(size);
      for (int idx = 0; idx < future_elements_.size(); idx++) {
        TF_RETURN_IF_ERROR(ReadElement(ctx, reader, idx, kFutureElements,
                                       &future_elements_[idx]));
      }
      return Status::OK();
    }

    // Used for coordination between the main thread, the runner thread, and
    // the worker threads.
    const std::shared_ptr<mutex> mu_;

    // Used for coordination between the main thread, the manager threads, and
    // the threadpool threads. In particular, the managers thread should only
    // schedule new calls into the threadpool when the number of in-flight
    // calls is less than the user specified level of parallelism and there
    // are slots available in the element `results` buffer.
    const std::shared_ptr<condition_variable> cond_var_;

    // Identifies the maximum number of parallel calls.
    const std::shared_ptr<model::SharedState> num_parallel_calls_;

    // Determines whether outputs can be produced in non-deterministic order.
    const bool sloppy_;

    // Iterator for input elements.
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(*mu_);

    // Identifies position in the interleave cycle.
    int64 block_index_ GUARDED_BY(*mu_) = 0;
    int64 cycle_index_ GUARDED_BY(*mu_) = 0;

    // Elements of the current interleave cycle.
    std::vector<std::shared_ptr<Element>> current_elements_ GUARDED_BY(*mu_);

    // Elements to be used in the interleave cycle in the future.
    std::deque<std::shared_ptr<Element>> future_elements_ GUARDED_BY(*mu_);

    // Identifies whether the global end of input has been reached.
    bool end_of_input_ GUARDED_BY(*mu_) = false;

    // Identifies the number of open iterators.
    int64 num_open_ GUARDED_BY(*mu_) = 0;

    // Identifies the number of outstanding calls for CurrentElementsManager.
    int64 current_num_calls_ GUARDED_BY(*mu_) = 0;
    // Identifies the number of outstanding calls for FutureElementsManager.
    int64 future_num_calls_ GUARDED_BY(*mu_) = 0;

    std::unique_ptr<thread::ThreadPool> thread_pool_;
    std::unique_ptr<Thread> current_elements_manager_ GUARDED_BY(*mu_);
    std::unique_ptr<Thread> future_elements_manager_ GUARDED_BY(*mu_);
    int64 element_id_counter_ GUARDED_BY(*mu_) = 0;

    // Identifies whether background threads should be cancelled.
    bool cancelled_ GUARDED_BY(*mu_) = false;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const int64 cycle_length_;
  const int64 block_length_;
  const int64 num_parallel_calls_;
  const int op_version_ = 2;
  const bool sloppy_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

ParallelInterleaveDatasetOp::ParallelInterleaveDatasetOp(
    OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  FunctionMetadata::Params params;
  params.is_multi_device_function = true;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSloppy, &sloppy_));
}

void ParallelInterleaveDatasetOp::MakeDataset(OpKernelContext* ctx,
                                              DatasetBase* input,
                                              DatasetBase** output) {
  int64 cycle_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCycleLength, &cycle_length));
  if (cycle_length == model::kAutotune) {
    cycle_length = port::NumSchedulableCPUs();
  }
  OP_REQUIRES(ctx, cycle_length > 0,
              errors::InvalidArgument("`cycle_length` must be > 0"));

  int64 block_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBlockLength, &block_length));
  OP_REQUIRES(ctx, block_length > 0,
              errors::InvalidArgument("`block_length` must be > 0"));

  int64 num_parallel_calls = 0;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  OP_REQUIRES(
      ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
      errors::InvalidArgument("num_parallel_calls must be greater than zero."));
  OP_REQUIRES(
      ctx, num_parallel_calls <= cycle_length,
      errors::InvalidArgument(
          "num_parallel_calls must less than or equal to cycle_length."));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  if (num_parallel_calls == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output = new Dataset(ctx, input, std::move(captured_func), cycle_length,
                        block_length, num_parallel_calls, sloppy_,
                        output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ParallelInterleaveDatasetV2").Device(DEVICE_CPU),
                        ParallelInterleaveDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("ParallelInterleaveDatasetV2");
}  // namespace
}  // namespace data
}  // namespace tensorflow

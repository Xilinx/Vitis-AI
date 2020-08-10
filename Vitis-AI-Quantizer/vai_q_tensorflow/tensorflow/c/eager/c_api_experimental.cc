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

#include "tensorflow/c/eager/c_api_experimental.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"

using tensorflow::string;

void TFE_OpConsumeInput(TFE_Op* op, TFE_TensorHandle* h, TF_Status* status) {
  op->operation.ConsumeInput(h->handle);
}

TFE_Profiler* TFE_NewProfiler() { return new TFE_Profiler(); }

bool TFE_ProfilerIsOk(TFE_Profiler* profiler) {
  return profiler->profiler->Status().ok();
}

void TFE_DeleteProfiler(TFE_Profiler* profiler) { delete profiler; }

void TFE_ProfilerSerializeToString(TFE_Profiler* profiler, TF_Buffer* buf,
                                   TF_Status* status) {
  string content;
  status->status = profiler->profiler->SerializeToString(&content);
  void* data = tensorflow::port::Malloc(content.length());
  content.copy(static_cast<char*>(data), content.length(), 0);
  buf->data = data;
  buf->length = content.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}

void TFE_StartProfilerServer(int port) {
  // Release child thread intentionally. The child thread can be terminated by
  // terminating the main thread.
  tensorflow::StartProfilerServer(port).release();
}

void TFE_ContextEnableGraphCollection(TFE_Context* ctx) {
  ctx->context->SetShouldStoreGraphs(true);
}

void TFE_ContextDisableGraphCollection(TFE_Context* ctx) {
  ctx->context->SetShouldStoreGraphs(false);
}

bool TFE_ProfilerClientStartTracing(const char* service_addr,
                                    const char* logdir, const char* worker_list,
                                    bool include_dataset_ops, int duration_ms,
                                    int num_tracing_attempts,
                                    TF_Status* status) {
  tensorflow::Status s =
      tensorflow::profiler::client::ValidateHostPortPair(service_addr);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return false;
  }
  s = tensorflow::profiler::client::StartTracing(
      service_addr, logdir, worker_list, include_dataset_ops, duration_ms,
      num_tracing_attempts);
  tensorflow::Set_TF_Status_from_Status(status, s);
  return s.ok();
}

void TFE_ProfilerClientMonitor(const char* service_addr, int duration_ms,
                               int monitoring_level, bool display_timestamp,
                               TF_Buffer* result, TF_Status* status) {
  tensorflow::Status s =
      tensorflow::profiler::client::ValidateHostPortPair(service_addr);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return;
  }
  string content;
  s = tensorflow::profiler::client::Monitor(
      service_addr, duration_ms, monitoring_level, display_timestamp, &content);
  void* data = tensorflow::port::Malloc(content.length());
  content.copy(static_cast<char*>(data), content.length(), 0);
  result->data = data;
  result->length = content.length();
  result->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
  tensorflow::Set_TF_Status_from_Status(status, s);
}

void TFE_MonitoringCounterCellIncrementBy(TFE_MonitoringCounterCell* cell,
                                          int64_t value) {
  cell->cell.IncrementBy(value);
}

int64_t TFE_MonitoringCounterCellValue(TFE_MonitoringCounterCell* cell) {
  return cell->cell.value();
}

TFE_MonitoringCounter0* TFE_MonitoringNewCounter0(const char* name,
                                                  TF_Status* status,
                                                  const char* description) {
  auto* result = new TFE_MonitoringCounter0({name, description});
  Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter0(TFE_MonitoringCounter0* counter) {
  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter0(
    TFE_MonitoringCounter0* counter) {
  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell()));
}

TFE_MonitoringCounter1* TFE_MonitoringNewCounter1(const char* name,
                                                  TF_Status* status,
                                                  const char* description,
                                                  const char* label1) {
  auto* result = new TFE_MonitoringCounter1({name, description, label1});
  Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter1(TFE_MonitoringCounter1* counter) {
  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter1(
    TFE_MonitoringCounter1* counter, const char* label1) {
  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell(label1)));
}

TFE_MonitoringCounter2* TFE_MonitoringNewCounter2(const char* name,
                                                  TF_Status* status,
                                                  const char* description,
                                                  const char* label1,
                                                  const char* label2) {
  auto* result =
      new TFE_MonitoringCounter2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter2(TFE_MonitoringCounter2* counter) {
  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter2(
    TFE_MonitoringCounter2* counter, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell(label1, label2)));
}

void TFE_MonitoringIntGaugeCellSet(TFE_MonitoringIntGaugeCell* cell,
                                   int64_t value) {
  cell->cell.Set(value);
}

int64_t TFE_MonitoringIntGaugeCellValue(TFE_MonitoringIntGaugeCell* cell) {
  return cell->cell.value();
}

TFE_MonitoringIntGauge0* TFE_MonitoringNewIntGauge0(const char* name,
                                                    TF_Status* status,
                                                    const char* description) {
  auto* result = new TFE_MonitoringIntGauge0({name, description});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge0(TFE_MonitoringIntGauge0* gauge) {
  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge0(
    TFE_MonitoringIntGauge0* gauge) {
  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringIntGauge1* TFE_MonitoringNewIntGauge1(const char* name,
                                                    TF_Status* status,
                                                    const char* description,
                                                    const char* label1) {
  auto* result = new TFE_MonitoringIntGauge1({name, description, label1});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge1(TFE_MonitoringIntGauge1* gauge) {
  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge1(
    TFE_MonitoringIntGauge1* gauge, const char* label1) {
  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringIntGauge2* TFE_MonitoringNewIntGauge2(const char* name,
                                                    TF_Status* status,
                                                    const char* description,
                                                    const char* label1,
                                                    const char* label2) {
  auto* result =
      new TFE_MonitoringIntGauge2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge2(TFE_MonitoringIntGauge2* gauge) {
  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge2(
    TFE_MonitoringIntGauge2* gauge, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

void TFE_MonitoringStringGaugeCellSet(TFE_MonitoringStringGaugeCell* cell,
                                      const char* value) {
  cell->cell.Set({value});
}

const void TFE_MonitoringStringGaugeCellValue(
    TFE_MonitoringStringGaugeCell* cell, TF_Buffer* buf) {
  tensorflow::string value = cell->cell.value();
  void* data = tensorflow::port::Malloc(value.length());
  value.copy(static_cast<char*>(data), value.length(), 0);
  buf->data = data;
  buf->length = value.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}

TFE_MonitoringStringGauge0* TFE_MonitoringNewStringGauge0(
    const char* name, TF_Status* status, const char* description) {
  auto* result = new TFE_MonitoringStringGauge0({name, description});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge0(TFE_MonitoringStringGauge0* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge0(
    TFE_MonitoringStringGauge0* gauge) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringStringGauge1* TFE_MonitoringNewStringGauge1(
    const char* name, TF_Status* status, const char* description,
    const char* label1) {
  auto* result = new TFE_MonitoringStringGauge1({name, description, label1});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge1(TFE_MonitoringStringGauge1* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge1(
    TFE_MonitoringStringGauge1* gauge, const char* label1) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringStringGauge2* TFE_MonitoringNewStringGauge2(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2) {
  auto* result =
      new TFE_MonitoringStringGauge2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge2(TFE_MonitoringStringGauge2* gauge) {
  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge2(
    TFE_MonitoringStringGauge2* gauge, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

void TFE_MonitoringBoolGaugeCellSet(TFE_MonitoringBoolGaugeCell* cell,
                                    bool value) {
  cell->cell.Set(value);
}

bool TFE_MonitoringBoolGaugeCellValue(TFE_MonitoringBoolGaugeCell* cell) {
  return cell->cell.value();
}

TFE_MonitoringBoolGauge0* TFE_MonitoringNewBoolGauge0(const char* name,
                                                      TF_Status* status,
                                                      const char* description) {
  auto* result = new TFE_MonitoringBoolGauge0({name, description});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge0(TFE_MonitoringBoolGauge0* gauge) {
  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge0(
    TFE_MonitoringBoolGauge0* gauge) {
  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringBoolGauge1* TFE_MonitoringNewBoolGauge1(const char* name,
                                                      TF_Status* status,
                                                      const char* description,
                                                      const char* label1) {
  auto* result = new TFE_MonitoringBoolGauge1({name, description, label1});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge1(TFE_MonitoringBoolGauge1* gauge) {
  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge1(
    TFE_MonitoringBoolGauge1* gauge, const char* label1) {
  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringBoolGauge2* TFE_MonitoringNewBoolGauge2(const char* name,
                                                      TF_Status* status,
                                                      const char* description,
                                                      const char* label1,
                                                      const char* label2) {
  auto* result =
      new TFE_MonitoringBoolGauge2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge2(TFE_MonitoringBoolGauge2* gauge) {
  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge2(
    TFE_MonitoringBoolGauge2* gauge, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

void TFE_MonitoringSamplerCellAdd(TFE_MonitoringSamplerCell* cell,
                                  double value) {
  cell->cell.Add(value);
}

void TFE_MonitoringSamplerCellValue(TFE_MonitoringSamplerCell* cell,
                                    TF_Buffer* buf) {
  string content;
  cell->cell.value().SerializeToString(&content);
  void* data = tensorflow::port::Malloc(content.length());
  content.copy(static_cast<char*>(data), content.length(), 0);
  buf->data = data;
  buf->length = content.length();
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
}

TFE_MonitoringBuckets* TFE_MonitoringNewExponentialBuckets(double scale,
                                                           double growth_factor,
                                                           int bucket_count) {
  return new TFE_MonitoringBuckets([scale, growth_factor, bucket_count]() {
    return tensorflow::monitoring::Buckets::Exponential(scale, growth_factor,
                                                        bucket_count);
  });
}

void TFE_MonitoringDeleteBuckets(TFE_MonitoringBuckets* buckets) {
  delete buckets;
}

TFE_MonitoringSampler0* TFE_MonitoringNewSampler0(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description) {
  auto* result = new TFE_MonitoringSampler0(
      {name, buckets->create_buckets(), description});
  Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler0(TFE_MonitoringSampler0* sampler) {
  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler0(
    TFE_MonitoringSampler0* sampler) {
  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell()));
}

TFE_MonitoringSampler1* TFE_MonitoringNewSampler1(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description, const char* label1) {
  auto* result = new TFE_MonitoringSampler1(
      {name, buckets->create_buckets(), description, label1});
  Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler1(TFE_MonitoringSampler1* sampler) {
  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler1(
    TFE_MonitoringSampler1* sampler, const char* label1) {
  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell(label1)));
}

TFE_MonitoringSampler2* TFE_MonitoringNewSampler2(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description, const char* label1, const char* label2) {
  auto* result = new TFE_MonitoringSampler2(
      {name, buckets->create_buckets(), description, label1, label2});
  Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler2(TFE_MonitoringSampler2* sampler) {
  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler2(
    TFE_MonitoringSampler2* sampler, const char* label1, const char* label2) {
  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell(label1, label2)));
}

void TFE_ContextOptionsSetMirroringPolicy(TFE_ContextOptions* options,
                                          TFE_ContextMirroringPolicy policy) {
  options->mirroring_policy = policy;
}

void TFE_ContextSetThreadLocalMirroringPolicy(
    TFE_Context* ctx, TFE_ContextMirroringPolicy policy) {
  ctx->context->SetThreadLocalMirroringPolicy(
      static_cast<tensorflow::ContextMirroringPolicy>(policy));
}

// Note: this function looks up a thread local policy. So it should be called in
// the appropriate client thread. In particular, in async mode, it may not be
// safe to call this function from the async EagerExecutor threads.
extern TFE_ContextMirroringPolicy TFE_ContextGetMirroringPolicy(
    TFE_Context* ctx) {
  return static_cast<TFE_ContextMirroringPolicy>(
      ctx->context->GetMirroringPolicy());
}

TFE_CancellationManager* TFE_NewCancellationManager() {
  return new TFE_CancellationManager;
}

void TFE_CancellationManagerStartCancel(
    TFE_CancellationManager* cancellation_manager) {
  cancellation_manager->cancellation_manager.StartCancel();
}

bool TFE_CancellationManagerIsCancelled(
    TFE_CancellationManager* cancellation_manager) {
  return cancellation_manager->cancellation_manager.IsCancelled();
}

void TFE_DeleteCancellationManager(
    TFE_CancellationManager* cancellation_manager) {
  delete cancellation_manager;
}

void TFE_OpSetCancellationManager(TFE_Op* op,
                                  TFE_CancellationManager* cancellation_manager,
                                  TF_Status* status) {
  op->operation.SetCancellationManager(
      &cancellation_manager->cancellation_manager);
}

TFE_Executor* TFE_NewExecutor(bool is_async) {
  return new TFE_Executor(is_async);
}

void TFE_DeleteExecutor(TFE_Executor* executor) { delete executor; }

bool TFE_ExecutorIsAsync(TFE_Executor* executor) {
  return executor->executor()->Async();
}

void TFE_ExecutorWaitForAllPendingNodes(TFE_Executor* executor,
                                        TF_Status* status) {
  status->status = executor->executor()->WaitForAllPendingNodes();
}

void TFE_ExecutorClearError(TFE_Executor* executor) {
  executor->executor()->ClearError();
}

void TFE_ContextSetExecutorForThread(TFE_Context* ctx, TFE_Executor* executor) {
  ctx->context->SetExecutorForThread(executor->executor());
}

TFE_Executor* TFE_ContextGetExecutorForThread(TFE_Context* ctx) {
  return new TFE_Executor(ctx->context->Executor());
}

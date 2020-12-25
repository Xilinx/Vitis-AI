/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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
// GRPC client to perform on-demand profiling

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace profiler {
namespace client {

Status ValidateHostPortPair(const string& host_port);

// Collects one sample of monitoring profile and shows user-friendly metrics.
// If timestamp flag is true, timestamp will be displayed in "%H:%M:%S" format.
Status Monitor(const tensorflow::string& service_addr, int duration_ms,
               int monitoring_level, bool display_timestamp, string* result);

// Starts tracing on a single or multiple hosts and saves the result in the
// given logdir. If no trace was collected, retries tracing for
// num_tracing_attempts.
Status StartTracing(const tensorflow::string& service_addr,
                    const tensorflow::string& logdir,
                    const tensorflow::string& workers_list,
                    bool include_dataset_ops, int duration_ms,
                    int num_tracing_attempts);

}  // namespace client
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_

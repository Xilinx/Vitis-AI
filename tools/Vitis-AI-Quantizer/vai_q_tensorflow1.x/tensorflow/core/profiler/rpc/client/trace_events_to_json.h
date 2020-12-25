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

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_TRACE_EVENTS_TO_JSON_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_TRACE_EVENTS_TO_JSON_H_

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/trace_events.pb.h"

namespace tensorflow {

namespace profiler {
namespace client {

// Converts trace events in the trace proto to a JSON string that can be
// consumed by catapult trace viewer.
string TraceEventsToJson(const Trace &trace);

}  // namespace client
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_TRACE_EVENTS_TO_JSON_H_

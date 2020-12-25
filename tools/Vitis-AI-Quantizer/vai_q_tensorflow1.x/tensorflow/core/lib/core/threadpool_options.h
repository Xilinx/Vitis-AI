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

#ifndef TENSORFLOW_CORE_LIB_CORE_THREADPOOL_OPTIONS_H_
#define TENSORFLOW_CORE_LIB_CORE_THREADPOOL_OPTIONS_H_

#include "tensorflow/core/lib/core/threadpool_interface.h"

namespace tensorflow {
namespace thread {

struct ThreadPoolOptions {
  // If not null, use this threadpool to schedule inter-op operation
  thread::ThreadPoolInterface* inter_op_threadpool;

  // If not null, use this threadpool to schedule intra-op operation
  thread::ThreadPoolInterface* intra_op_threadpool;
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_THREADPOOL_OPTIONS_H_

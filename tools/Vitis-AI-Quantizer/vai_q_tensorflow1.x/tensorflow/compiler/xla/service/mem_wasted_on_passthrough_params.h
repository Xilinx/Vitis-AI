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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEM_WASTED_ON_PASSTHROUGH_PARAMS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEM_WASTED_ON_PASSTHROUGH_PARAMS_H_

#include <memory>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// This pass LOGs memory waste due to pass-through params, i.e. output params
// that are just an input argument. These arguments are currently copied from
// the input buffer to a newly allocated output buffer. See b/133276457 for
// more details.
//
// This pass does not modify the HLO.
class MemWastedOnPassthroughParams : public HloModulePass {
 public:
  MemWastedOnPassthroughParams() = default;
  ~MemWastedOnPassthroughParams() override = default;
  absl::string_view name() const override {
    return "mem_wasted_on_passthrough_params";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEM_WASTED_ON_PASSTHROUGH_PARAMS_H_

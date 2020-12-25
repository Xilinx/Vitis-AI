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

#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"

#include <unordered_set>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/rename.h"

namespace tflite {
namespace gpu {
namespace gl {

Status MergeCode(CompiledNodeAttributes* attr,
                 CompiledNodeAttributes* merged_attr) {
  // build a map of known names.
  std::unordered_set<std::string> known_names;
  for (const auto& parameter : merged_attr->code.parameters) {
    known_names.insert(parameter.name);
  }
  for (const auto& object : merged_attr->code.objects) {
    known_names.insert(object.first);
  }

  // Rewrite parameters with unique names.
  int index =
      merged_attr->code.parameters.size() + merged_attr->code.objects.size();
  RETURN_IF_ERROR(Rename(
      [&](absl::string_view name) -> std::string {
        std::string n(name.begin(), name.end());
        // if a name is unique, then keep it as is. Otherwise append an unique
        // index.
        if (known_names.find(n) == known_names.end()) {
          return n;
        }
        return absl::StrCat(n, index++);
      },
      &attr->code));
  std::move(attr->code.objects.begin(), attr->code.objects.end(),
            std::back_inserter(merged_attr->code.objects));
  std::move(attr->code.parameters.begin(), attr->code.parameters.end(),
            std::back_inserter(merged_attr->code.parameters));
  std::move(attr->node_indices.begin(), attr->node_indices.end(),
            std::back_inserter(merged_attr->node_indices));
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

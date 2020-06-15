/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_KNOWN_PATTERNS_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_KNOWN_PATTERNS_H_

#include "tensorflow/contrib/decent_q/utils/transform_utils.h"

namespace tensorflow {
namespace decent_q {

// Quantizable op types
extern const std::set<string> quantizable_op_types;

// Supported op types: not quantizable but can be handled by decent
extern const std::set<string> supported_op_types;

// Unsupported op types: not quantizable and cannot be handled by decent
extern const std::set<string> unsupported_op_types;

// Known patterns
extern const std::vector<std::tuple<const string, const OpTypePattern>>
    known_patterns;

// Known ignore patterns
extern const std::vector<std::tuple<const string, const OpTypePattern>>
    known_ignore_patterns;

// Patterns with computation
extern const std::set<string> compute_patterns;

// Get pattern name from id
const string get_pattern_name_from_id(const int pattern_id);

// Get ignore pattern name from id
const string get_ignore_pattern_name_from_id(const int pattern_id);

// Get input nodes from pattern
std::vector<const NodeDef*> get_input_nodes(const NodeMatch& match,
                                            const string& pattern_name);

// Get ignore nodes from pattern
std::vector<const NodeDef*> get_ignore_nodes(const NodeMatch& match,
                                             const string& pattern_name);

}  // namespace decent_q
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_KNOWN_PATTERNS_H_

/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xir/op/op_def.hpp"
#include "xir/op/shape_inference.hpp"

XIR_REGISTER_OPS(XIR_OP("TestContrib",
                        XIR_MAKE_VEC(XIR_OP_ARG("input", REQUIRED, FLOAT,
                                                "input, 3d-[w, h, ch]")),
                        XIR_MAKE_VEC(), [](xir::Op* op) {}, "TestContrib"));
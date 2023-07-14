/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#pragma once

#include <map>

#include "vart/runner.hpp"
#include "vitis/ai/erl_msg_box.hpp"
#include "vitis/ai/thread_pool.hpp"
namespace vart {
using init_function_t = vart::Runner* (*)(const xir::Subgraph*, xir::Attrs*);
std::unique_ptr<Runner> create_async_runner(init_function_t f,
                                            const xir::Subgraph* subgraph,
                                            xir::Attrs* attrs);

}  // namespace vart

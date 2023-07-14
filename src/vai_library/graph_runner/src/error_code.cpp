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

#include "UniLog/ErrorCode.hpp"

REGISTER_ERROR_CODE(VAILIB_GRAPH_RUNNER_NOT_FIND,
                    "GraphTask can not find tensor or tensor buffer!", "");
REGISTER_ERROR_CODE(VAILIB_GRAPH_RUNNER_DPU_BATCH_ERROR,
                    "GraphTask get dpu batch not equal!", "");
REGISTER_ERROR_CODE(VAILIB_GRAPH_RUNNER_NOT_SUPPORT,
                    "The function or value are not supported in graph runner!",
                    "");
REGISTER_ERROR_CODE(VAILIB_GRAPH_RUNNER_NOT_OVERRIDE,
                    "The funtion has not been overridden! ", "");

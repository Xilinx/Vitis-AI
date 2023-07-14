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

REGISTER_ERROR_CODE(VAILIB_CPU_RUNNER_OPEN_LIB_ERROR,
                    "dlopen can not open lib!", "");
REGISTER_ERROR_CODE(VAILIB_CPU_RUNNER_LOAD_LIB_SYM_ERROR,
                    "dlsym load symbol error!", "");
REGISTER_ERROR_CODE(VAILIB_CPU_RUNNER_TENSOR_BUFFER_NOT_FIND,
                    "Can not find tensor buffer with this name!", "");
REGISTER_ERROR_CODE(VAILIB_CPU_RUNNER_TENSOR_BUFFER_NOT_CONTINOUS,
                    "Tensor buffer not continous!", "");
REGISTER_ERROR_CODE(VAILIB_CPU_RUNNER_READ_FILE_ERROR, "Fail to read file!",
                    "");
REGISTER_ERROR_CODE(VAILIB_CPU_RUNNER_WRITE_FILE_ERROR, "Fail to write file!",
                    "");
REGISTER_ERROR_CODE(VAILIB_CPU_RUNNER_CPU_OP_NOT_FIND,
                    "Can not find op with this name!", "");


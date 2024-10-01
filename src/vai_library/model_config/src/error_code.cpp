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

REGISTER_ERROR_CODE(VAILIB_MODEL_CONFIG_NOT_FIND, "Model config info not find!",
                    "");
REGISTER_ERROR_CODE(VAILIB_MODEL_CONFIG_OPEN_ERROR,
                    "Model config file or directory open error!", "");
REGISTER_ERROR_CODE(VAILIB_MODEL_CONFIG_CONFIG_PARSE_ERROR,
                    "Model config file parse error!", "");


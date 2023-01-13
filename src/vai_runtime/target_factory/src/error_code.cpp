/*
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "UniLog/ErrorCode.hpp"

REGISTER_ERROR_CODE(TARGET_FACTORY_MULTI_REGISTERED_TARGET,
                    "Multiple registration of target!", "");
REGISTER_ERROR_CODE(TARGET_FACTORY_UNREGISTERED_TARGET, "Unregistered target!",
                    "");

REGISTER_ERROR_CODE(TARGET_FACTORY_INVALID_TYPE, "Invalid target type!", "");
REGISTER_ERROR_CODE(TARGET_FACTORY_INVALID_ISA_VERSION,
                    "Invalid target ISA version!", "");
REGISTER_ERROR_CODE(TARGET_FACTORY_INVALID_FEATURE_CODE,
                    "Invalid target feature code!", "");
REGISTER_ERROR_CODE(TARGET_FACTORY_INVALID_ARCH,
                    "Invalid target arch!", "");

REGISTER_ERROR_CODE(TARGET_FACTORY_PARSE_TARGET_FAIL,
                    "Fail to parse target from prototxt!", "");

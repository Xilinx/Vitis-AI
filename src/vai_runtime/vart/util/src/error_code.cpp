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

#include <UniLog/ErrorCode.hpp>

REGISTER_ERROR_CODE(VART_OPEN_DEVICE_FAIL, "Cannot open device", "");
REGISTER_ERROR_CODE(VART_LOAD_XCLBIN_FAIL, "Bitstream download failed!", "");
REGISTER_ERROR_CODE(VART_LOCK_DEVICE_FAIL, "Cannot lock device!", "");

REGISTER_ERROR_CODE(VART_FUNC_NOT_SUPPORT, "Function not support!", "");
REGISTER_ERROR_CODE(VART_XMODEL_ERROR, "Xmodel error!", "");
REGISTER_ERROR_CODE(VART_GRAPH_ERROR, "Graph error!", "");
REGISTER_ERROR_CODE(VART_TENSOR_INFO_ERROR, "Tensor info error!", "");
REGISTER_ERROR_CODE(VART_DPU_INFO_ERROR, "DPU info error!", "");
REGISTER_ERROR_CODE(VART_SYSTEM_ERROR, "File system error!", "");
REGISTER_ERROR_CODE(VART_DEVICE_BUSY, "Device busy!", "");
REGISTER_ERROR_CODE(VART_DEVICE_MISMATCH, "Device mismatch!", "");

REGISTER_ERROR_CODE(VART_DPU_ALLOC_ERROR, "DPU allocate error!", "");
REGISTER_ERROR_CODE(VART_VERSION_MISMATCH, "Version mismatch!", "");

REGISTER_ERROR_CODE(VART_OUT_OF_RANGE, "Array index out of range!", "");
REGISTER_ERROR_CODE(VART_SIZE_MISMATCH, "Array size not match!", "");

REGISTER_ERROR_CODE(VART_XRT_NULL_PTR, "Nullptr!", "");
REGISTER_ERROR_CODE(VART_XRT_DEVICE_BUSY, "Device busy!", "");
REGISTER_ERROR_CODE(VART_XRT_READ_ERROR, "Read error!", "");
REGISTER_ERROR_CODE(VART_XRT_READ_CU_ERROR, "Read cu fatal!", "");
REGISTER_ERROR_CODE(VART_XRT_FUNC_FAULT, "XRT function fault!", "");

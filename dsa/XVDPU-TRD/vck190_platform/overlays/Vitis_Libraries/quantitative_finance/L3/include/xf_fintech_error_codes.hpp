/*
 * Copyright 2019 Xilinx, Inc.
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

#ifndef _XF_FINTECH_ERROR_CODES_H_
#define _XF_FINTECH_ERROR_CODES_H_

#define XLNX_OK (0x00000000)

#define XLNX_ERROR_DEVICE_OWNED_BY_ANOTHER_OCL_CONTROLLER (0x00000001)
#define XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER (0x00000002)
#define XLNX_ERROR_DEVICE_INVALID_OCL_CONTROLLER (0x00000003)
#define XLNX_ERROR_DEVICE_ALREADY_OWNED_BY_THIS_OCL_CONTROLLER (0x00000004)

#define XLNX_ERROR_OCL_CONTROLLER_ALREADY_OWNS_ANOTHER_DEVICE (0x00000005)
#define XLNX_ERROR_OCL_CONTROLLER_DOES_NOT_OWN_ANY_DEVICE (0x00000006)

#define XLNX_ERROR_FAILED_TO_IMPORT_XCLBIN_FILE (0x00000007)

#define XLNX_ERROR_OPENCL_CALL_ERROR (0x00000008)

#define XLNX_ERROR_NOT_SUPPORTED (0x00000009)

#define XLNX_ERROR_MODEL_INTERNAL_ERROR (0x0000000A)

#define XLNX_ERROR_LINEAR_INTERPOLATION_FAILED (0x0000000B)

#endif //_XF_FINTECH_ERROR_CODES_H_
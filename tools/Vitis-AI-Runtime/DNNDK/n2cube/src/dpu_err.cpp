/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <pthread.h>

#include "dpu_err.h"
#include "dnndk/n2cube.h"

#ifdef __cplusplus
extern "C" {
#endif
static int gN2cubeLogMode = N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT;

void dpuInitExceptionMode(void)
{
    if ((N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT != gN2cubeLogMode)
        && (N2CUBE_EXCEPTION_MODE_RET_ERR_CODE != gN2cubeLogMode)) {
        gN2cubeLogMode = N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT;
    }
}

int dpuSetExceptionMode(int mode)
{
    if ((N2CUBE_EXCEPTION_MODE_RET_ERR_CODE != mode)
        && (N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT != mode))
    {
        return N2CUBE_ERR_PARAM_VALUE;
    }

    gN2cubeLogMode = mode;
    return N2CUBE_OK;
}

int dpuGetExceptionMode()
{
    return gN2cubeLogMode;
}


#ifdef __cplusplus
}
#endif

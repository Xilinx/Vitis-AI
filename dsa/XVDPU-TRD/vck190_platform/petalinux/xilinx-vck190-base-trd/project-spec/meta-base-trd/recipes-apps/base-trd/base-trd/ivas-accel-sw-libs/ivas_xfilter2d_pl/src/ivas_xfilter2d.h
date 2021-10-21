/*
 * Copyright 2020 - 2021 Xilinx, Inc.
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

#ifndef __IVAS_XFILTER2D_H__
#define __IVAS_XFILTER2D_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <ivas/ivas_kernel.h>
#include <string.h>

#define IVAS_FOURCC_YUY2 0x56595559

typedef enum
{
    IVAS_FILTER2D_PRESET_BLUR,
    IVAS_FILTER2D_PRESET_EDGE,
    IVAS_FILTER2D_PRESET_HEDGE,
    IVAS_FILTER2D_PRESET_VEDGE,
    IVAS_FILTER2D_PRESET_EMBOSS,
    IVAS_FILTER2D_PRESET_HGRAD,
    IVAS_FILTER2D_PRESET_VGRAD,
    IVAS_FILTER2D_PRESET_IDENTITY,
    IVAS_FILTER2D_PRESET_SHARPEN,
    IVAS_FILTER2D_PRESET_HSOBEL,
    IVAS_FILTER2D_PRESET_VSOBEL,
} IVASFilter2dFilterPreset;

typedef struct {
    int value;
    const char *nick;
} EnumValue;

static const EnumValue filter_presets[] = {
    {IVAS_FILTER2D_PRESET_BLUR, "blur"},
    {IVAS_FILTER2D_PRESET_EDGE, "edge"},
    {IVAS_FILTER2D_PRESET_HEDGE, "horizontal edge"},
    {IVAS_FILTER2D_PRESET_VEDGE, "vertical edge"},
    {IVAS_FILTER2D_PRESET_EMBOSS, "emboss"},
    {IVAS_FILTER2D_PRESET_HGRAD, "horizontal gradient"},
    {IVAS_FILTER2D_PRESET_VGRAD, "vertical gradient"},
    {IVAS_FILTER2D_PRESET_IDENTITY, "identity"},
    {IVAS_FILTER2D_PRESET_SHARPEN, "sharpen"},
    {IVAS_FILTER2D_PRESET_HSOBEL, "horizontal sobel"},
    {IVAS_FILTER2D_PRESET_VSOBEL, "vertical sobel"},
    {0, NULL}
};

#define KSIZE 3
typedef short int coeff_t[KSIZE][KSIZE];

static const coeff_t coeffs[] = {
    [IVAS_FILTER2D_PRESET_BLUR] = {
        {1, 1, 1},
        {1, -7, 1},
        {1, 1, 1}
    },
    [IVAS_FILTER2D_PRESET_EDGE] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    },
    [IVAS_FILTER2D_PRESET_HEDGE] = {
        {0, -1, 0},
        {0, 2, 0},
        {0, -1, 0}
    },
    [IVAS_FILTER2D_PRESET_VEDGE] = {
        {0, 0, 0},
        {-1, 2, -1},
        {0, 0, 0}
    },
    [IVAS_FILTER2D_PRESET_EMBOSS] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    },
    [IVAS_FILTER2D_PRESET_HGRAD] = {
        {-1, -1, -1},
        {0, 0, 0},
        {1, 1, 1}
    },
    [IVAS_FILTER2D_PRESET_VGRAD] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    },
    [IVAS_FILTER2D_PRESET_IDENTITY] = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    },
    [IVAS_FILTER2D_PRESET_SHARPEN] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    },
    [IVAS_FILTER2D_PRESET_HSOBEL] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    },
    [IVAS_FILTER2D_PRESET_VSOBEL] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    }
};

typedef struct _kern_priv
{
    int log_level;
    uint32_t in_fourcc;
    uint32_t out_fourcc;
    const char *filter_preset;
    IVASFrame *params;
} Filter2dKernelPriv;

static inline const coeff_t *get_coeff_by_preset(const char *preset) {
    int i = 0;
    while (filter_presets[i].nick != NULL) {
        if (strcmp(preset, filter_presets[i].nick) == 0)
            return &coeffs[filter_presets[i].value];
        i++;
    }
    /* return identity if preset not matched */
    return &coeffs[IVAS_FILTER2D_PRESET_IDENTITY];
}

#define eos(s) ((s)+strlen(s))
static inline void coeff_to_str(char *s, const coeff_t c) {
    s[0] = '\0';
    sprintf(eos(s), "coeffs = ");
    for (int i=0; i<KSIZE; i++) {
        sprintf(eos(s), "[ ");
        for (int j=0; j<KSIZE; j++) {
            sprintf(eos(s), "%d ", c[i][j]);
        }
        sprintf(eos(s), "] ");
    }
}

int32_t xlnx_kernel_init(IVASKernel *handle);
uint32_t xlnx_kernel_deinit(IVASKernel *handle);
int32_t xlnx_kernel_start(IVASKernel *handle, int start, \
    IVASFrame *input[MAX_NUM_OBJECT], IVASFrame *output[MAX_NUM_OBJECT]);
int32_t xlnx_kernel_done(IVASKernel *handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //__IVAS_XFILTER2D_H__

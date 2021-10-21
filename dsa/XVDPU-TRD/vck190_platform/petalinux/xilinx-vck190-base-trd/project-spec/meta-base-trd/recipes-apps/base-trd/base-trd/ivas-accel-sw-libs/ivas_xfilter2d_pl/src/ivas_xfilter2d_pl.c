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

#include <ivas/ivaslogs.h>

#include "ivas_xfilter2d.h"

uint32_t xlnx_kernel_deinit(IVASKernel *handle)
{
    Filter2dKernelPriv *kpriv = (Filter2dKernelPriv *) handle->kernel_priv;
    ivas_free_buffer(handle, kpriv->params);
    free(kpriv);
    return 0;
}

int32_t xlnx_kernel_init(IVASKernel *handle)
{
    json_t *jconfig = handle->kernel_config;
    json_t *val, *ival, *jval; /* kernel config from app */
    Filter2dKernelPriv *kpriv;
    const char *fourcc = NULL;
    char s[64] = "\0";
    const coeff_t *pcoeff = NULL;
    coeff_t coeff;
    size_t isize, jsize;

    kpriv = (Filter2dKernelPriv *)calloc(1, sizeof(Filter2dKernelPriv));
    if (!kpriv) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, 0,
            "Error: Unable to allocate filter2d kernel memory");
    }

    /* set multiprocess mode */
    handle->is_multiprocess = 1;

    /* parse json config */

    /* debug_level */
    val = json_object_get(jconfig, "debug_level");
    if (!val || !json_is_integer(val)) {
        kpriv->log_level = LOG_LEVEL_WARNING;
    } else {
        kpriv->log_level = json_integer_value(val);
    }
    LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
        "Log level set to %d", kpriv->log_level);

    /* in_fourcc */
    val = json_object_get(jconfig, "in_fourcc");
    if (!val || !json_is_string(val)) {
        kpriv->in_fourcc = 0;
    } else {
        fourcc = json_string_value(val);
        if (strcmp(fourcc, "YUY2") == 0)
            kpriv->in_fourcc = IVAS_FOURCC_YUY2;
        else
            kpriv->in_fourcc = 0;
    }
    LOG_MESSAGE (LOG_LEVEL_INFO, kpriv->log_level,
        "in_fourcc = %s (0x%08x)", fourcc, kpriv->in_fourcc);

    /* out_fourcc  */
    val = json_object_get(jconfig, "out_fourcc");
    if (!val || !json_is_string(val)) {
        kpriv->out_fourcc = 0;
    } else {
        fourcc = json_string_value(val);
        if (strcmp(fourcc, "YUY2") == 0)
            kpriv->out_fourcc = IVAS_FOURCC_YUY2;
        else
            kpriv->out_fourcc = 0;
    }
    LOG_MESSAGE (LOG_LEVEL_INFO, kpriv->log_level,
        "out_fourcc = %s (0x%08x)", fourcc, kpriv->out_fourcc);

    /* filter_preset */
    val = json_object_get(jconfig, "filter_preset");
    if (!val || !json_is_string(val)) {
        kpriv->filter_preset = strdup("identity");
        LOG_MESSAGE (LOG_LEVEL_WARNING, kpriv->log_level,
            "Unexpected value for filter_preset: using identity");
    } else {
        kpriv->filter_preset = json_string_value(val);
    }

    LOG_MESSAGE (LOG_LEVEL_INFO, kpriv->log_level,
        "preset = %s", kpriv->filter_preset);

    if (strcmp(kpriv->filter_preset, "custom") != 0) {
        pcoeff = get_coeff_by_preset(kpriv->filter_preset);
        LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
            "Loading coefficients from preset");
    } else {
        LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
            "Loading custom coefficients");

        /* filter_coefficients */
        val = json_object_get(jconfig, "filter_coefficients");
        if (!val || !json_is_array(val)) {
            LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                "Unexpected value for filter_coefficients");
            return -1;
        }

        isize = json_array_size(val);
        if (isize != KSIZE) {
            LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                "Unexpected value for filter_coefficients");
            return -1;
        }

        /* outer array */
        for (int i=0; i<isize; i++) {
            ival = json_array_get(val, i);
            if (!ival || !json_is_array(ival)) {
                LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                    "Unexpected value for filter_coefficients");
                return -1;
            }

            jsize = json_array_size(ival);
            if (jsize != KSIZE) {
                LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                    "Unexpected value for filter_coefficients");
                return -1;
            }

            /* inner array */
            for (int j=0; j<jsize; j++) {
                jval = json_array_get(ival, j);
                if (!jval || !json_is_integer(jval)) {
                    LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                        "Unexpected value for filter_coefficients");
                    return -1;
                }

                coeff[i][j] = json_integer_value(jval);
            }
        }
        pcoeff = &coeff;
    }

    coeff_to_str(s, *pcoeff);
    LOG_MESSAGE (LOG_LEVEL_INFO, kpriv->log_level, "%s", s);

    /* set coefficients */
    kpriv->params = ivas_alloc_buffer (handle, sizeof(*pcoeff),
        IVAS_INTERNAL_MEMORY, NULL);
    memcpy(kpriv->params->vaddr[0], *pcoeff, sizeof(*pcoeff));

    handle->kernel_priv = (void *) kpriv;

    return 0;
}

int32_t xlnx_kernel_start(IVASKernel *handle, int start,
    IVASFrame *input[MAX_NUM_OBJECT], IVASFrame *output[MAX_NUM_OBJECT])
{
    Filter2dKernelPriv *kpriv;
    kpriv = (Filter2dKernelPriv *) handle->kernel_priv;

    json_t *jconfig = handle->kernel_dyn_config;
    json_t *val, *ival, *jval;

    char s[64] = "\0";
    const coeff_t *pcoeff = NULL;
    coeff_t coeff;
    size_t isize, jsize;
    unsigned int err = 0;

    if (jconfig) {
        /* filter_preset */
        val = json_object_get(jconfig, "filter_preset");
        if (val && json_is_string(val)) {
            kpriv->filter_preset = json_string_value(val);
            pcoeff = get_coeff_by_preset(kpriv->filter_preset);
            LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
                "Set filter_preset = %s", kpriv->filter_preset);
        }

        /* filter_coefficients */
        val = json_object_get(jconfig, "filter_coefficients");
        if (val && json_is_array(val)) {
            isize = json_array_size(val);
            if (isize != KSIZE) {
                LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                    "Unexpected value for filter_coefficients");
                return -1;
            }

            /* outer array */
            for (int i=0; i<isize; i++) {
                ival = json_array_get(val, i);
                if (ival && json_is_array(ival)) {
                    jsize = json_array_size(ival);
                    if (jsize != KSIZE) {
                        LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                            "Unexpected value for filter_coefficients");
                        return -1;
                    }

                    /* inner array */
                    for (int j=0; j<jsize; j++) {
                        jval = json_array_get(ival, j);
                        if (jval && json_is_integer(jval)) {
                            coeff[i][j] = json_integer_value(jval);
                        } else {
                            LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                                "Unexpected value for filter_coefficients");
                            return -1;
                        }
                    }
                } else {
                    LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level,
                        "Unexpected value for filter_coefficients");
                    return -1;
                }
            }
            pcoeff = &coeff;
        }
    }

    if (pcoeff) {
        coeff_to_str(s, *pcoeff);
        LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level, "%s", s);

        /* set coefficients */
        memcpy(kpriv->params->vaddr[0], *pcoeff, sizeof(*pcoeff));
    }

    /* Input frame */
    ivas_register_write(handle, &(input[0]->paddr[0]), sizeof(uint64_t), 0x10);
    /* Output frame */
    ivas_register_write(handle, &(output[0]->paddr[0]), sizeof(uint64_t), 0x1c);
    /* Kernel Params (coefficients) */
    ivas_register_write(handle, &(kpriv->params->paddr[0]),
        sizeof(uint64_t), 0x28);
    /* height */
    ivas_register_write(handle, &(input[0]->props.height),
        sizeof(uint32_t), 0x34);
    /* width */
    ivas_register_write(handle, &(input[0]->props.width),
        sizeof(uint32_t), 0x3c);
    /* in_fourcc */
    ivas_register_write(handle, &(kpriv->in_fourcc),
        sizeof(uint32_t), 0x44);
    /* out_fourcc */
    ivas_register_write(handle, &(kpriv->out_fourcc),
        sizeof(uint32_t), 0x4c);

    /* start */
    return ivas_kernel_start(handle);
}

int32_t xlnx_kernel_done(IVASKernel *handle)
{
    return ivas_kernel_done(handle, 100);
}

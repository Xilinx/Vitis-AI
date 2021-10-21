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
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ivas_xfilter2d.h"

using namespace cv;
using namespace std;

extern "C"
{

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

    int yiloc = (IVAS_FOURCC_YUY2 == kpriv->in_fourcc) ? 0 : 1;
    int ciloc = (IVAS_FOURCC_YUY2 == kpriv->in_fourcc) ? 1 : 0;
    int yuyvout = (IVAS_FOURCC_YUY2 == kpriv->out_fourcc);

    IVASFrame *inframe = input[0];
    uint32_t height = inframe->props.height;
    uint32_t width = inframe->props.width;
    char *indata = (char *) inframe->vaddr[0];

    IVASFrame *outframe = output[0];
    char *outdata = (char *) outframe->vaddr[0];
    pcoeff = (const coeff_t *) kpriv->params->vaddr[0];

    Mat src (height, width, CV_8UC2, indata);
    Mat dst (height, width, CV_8UC2, outdata);

    // planes
    std::vector<Mat> iplanes;
    std::vector<Mat> oplanes;

    // convert kernel from short to int
    int coeff_i[KSIZE][KSIZE];
    for (int i = 0; i < KSIZE; i++) {
        for (int j = 0; j < KSIZE; j++) {
            coeff_i[i][j] = (*pcoeff)[i][j];
        }
    }
    Mat kernel = Mat (KSIZE, KSIZE, CV_32SC1, (int *) coeff_i);

    // anchor
    Point anchor = Point (-1, -1);

    // filter
    split (src, iplanes);
    filter2D (iplanes[yiloc], iplanes[yiloc], -1, kernel, anchor, 0,
        BORDER_DEFAULT);

    if (yuyvout) {
        oplanes.push_back (iplanes[yiloc]);
        oplanes.push_back (iplanes[ciloc]);
    } else {
        oplanes.push_back (iplanes[ciloc]);
        oplanes.push_back (iplanes[yiloc]);
    }
    merge (oplanes, dst);

    return 0;
}

int32_t xlnx_kernel_done(IVASKernel *handle)
{
  return 0;
}

} // extern "C"


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

#pragma once
#include <vart/vart.h>
#include <xir/cxir.h>
#ifdef __cplusplus
extern "C" {
#endif

vart_runner_t vai_lib_create_graph_runner(xir_graph_t graph, xir_attrs_t attrs);

#ifdef __cplusplus
}
#endif

/* Local Variables: */
/* mode:c */
/* c-basic-offset: 2 */
/* coding: undecided-unix */
/* End: */

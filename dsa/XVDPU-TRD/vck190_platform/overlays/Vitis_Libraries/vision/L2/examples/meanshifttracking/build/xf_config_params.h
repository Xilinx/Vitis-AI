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

#define INPUT_PTR_WIDTH 32

// Maximum supported objects for tracking
#define XF_MAX_OBJECTS 10

// set the maximum height and width from the objects given for latency report and resource allocation
#define XF_MAX_OBJ_HEIGHT 250
#define XF_MAX_OBJ_WIDTH 250

// maximum number of iterations for centroid convergence
#define XF_MAX_ITERS 4

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

#define SAD_WINDOW_SIZE 11

/* NO_OF_DISPARITIES must be greater than '0' and less than the image width */
#define NO_OF_DISPARITIES 32

/* NO_OF_DISPARITIES must not be lesser than PARALLEL_UNITS and NO_OF_DISPARITIES/PARALLEL_UNITS must be a
 * non-fractional number */
#define PARALLEL_UNITS 32

#define XF_USE_URAM false

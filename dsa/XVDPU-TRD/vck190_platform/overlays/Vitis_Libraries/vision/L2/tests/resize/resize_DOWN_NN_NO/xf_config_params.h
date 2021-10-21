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

#define RO 0 // Resource Optimized (8-pixel implementation)
#define NO 1 // Normal Operation (1-pixel implementation)
// port widths
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 128

// For Nearest Neighbor & Bilinear Interpolation, max down scale factor 2 for all 1-pixel modes, and for upscale in x
// direction
#define MAXDOWNSCALE 2

#define RGB 0
#define GRAY 1
/* Interpolation type*/
#define INTERPOLATION 0
// 0 - Nearest Neighbor Interpolation
// 1 - Bilinear Interpolation
// 2 - AREA Interpolation

/* Input image Dimensions */
#define WIDTH 3840  // Maximum Input image width
#define HEIGHT 2160 // Maximum Input image height

/* Output image Dimensions */
#define NEWWIDTH 1920  // Maximum output image width
#define NEWHEIGHT 1080 // Maximum output image height

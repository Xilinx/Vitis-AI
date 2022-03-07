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

#define RO 0 // 8 Pixel Processing
#define NO 1 // 1 Pixel Processing

// Number of rows in the input image
#define HEIGHT 2160
// Number of columns in  in the input image
#define WIDTH 3840

// Number of rows of input image to be stored
#define NUM_STORE_ROWS 100

// Number of rows of input image after which output image processing must start
#define START_PROC 50

#define RGBA 1
#define GRAY 0

// transform type 0-NN 1-BILINEAR
#define INTERPOLATION 0

// transform type 0-AFFINE 1-PERSPECTIVE
#define TRANSFORM_TYPE 1
#define XF_USE_URAM true

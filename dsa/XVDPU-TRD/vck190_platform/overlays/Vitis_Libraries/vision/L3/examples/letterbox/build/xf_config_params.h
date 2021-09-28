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

/* Input/output port width */
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 128

/* Parameter for Resize kernel */
#define WIDTH 1920    // Maximum Input image width
#define HEIGHT 1080   // Maximum Input image height
#define NEWWIDTH 720  // Maximum output image width
#define NEWHEIGHT 720 // Maximum output image height

#define MAXDOWNSCALE 4
#define RGB 1
#define GRAY 0
/* Interpolation type*/
#define INTERPOLATION 1
#define RO 1
#define NO 0

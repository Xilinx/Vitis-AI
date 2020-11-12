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

#define WIDTH 1920
#define HEIGHT 1080
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 128
#define T_CHANNELS 3
#define CPW 3
#define NPC_TEST 1
#define PACK_MODE 0
#define X_WIDTH 8
#define ALPHA_WIDTH 16
#define BETA_WIDTH 16
#define GAMMA_WIDTH 8
#define OUT_WIDTH 16

#define X_IBITS 8
#define ALPHA_IBITS 8
#define BETA_IBITS 4
#define GAMMA_IBITS 8
#define OUT_IBITS 9

#define SIGNED_IN 0

#define OPMODE 0

#define NEWWIDTH 608 // Maximum output image width
#define NEWHEIGHT 608

#define MAXDOWNSCALE 4

//#define RGB 1
//#define GRAY 0
/* Interpolation type*/
#define INTERPOLATION 1
#define BGR2RGB 1
#define RO 0
#define NO 1

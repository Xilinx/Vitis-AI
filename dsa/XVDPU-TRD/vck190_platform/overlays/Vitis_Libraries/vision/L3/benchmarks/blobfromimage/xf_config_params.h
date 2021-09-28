/*
 * Copyright 2020 Xilinx, Inc.
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

// Max image resoultion

// Enable or disable channel Swap
#define BGR2RGB 0
// Enable or disable crop
#define CROP 1

static constexpr int WIDTH = 1920;
static constexpr int HEIGHT = 1080;

static constexpr int INPUT_PTR_WIDTH = 128;
static constexpr int OUTPUT_PTR_WIDTH = 128;

static constexpr int IN_TYPE = XF_8UC3;
static constexpr int OUT_TYPE = XF_8UC3;
// Pixels processed per cycle
static constexpr int NPC = XF_NPPC1;

// preprocess kernel params out = (in - a) * b
// a, b and out are fixed point values and below params are used to configure
// the width and integer bits
static constexpr int WIDTH_A = 8;
static constexpr int IBITS_A = 8;
static constexpr int WIDTH_B = 8;
static constexpr int IBITS_B = 4; // so B is 8-bit wide and 4-bits are integer bits
static constexpr int WIDTH_OUT = 8;
static constexpr int IBITS_OUT = 8;

// Resize configuration parameters
static constexpr int NEWWIDTH = 300; // Maximum output image width
static constexpr int NEWHEIGHT = 300;

static constexpr int MAXDOWNSCALE = 9;

static constexpr int INTERPOLATION = 1;

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
#pragma once
#define ORDER 8
#define RTM_x 50
#define RTM_y 60
#define RTM_z 80
#define NTime 9
#define MaxB 10
#define MaxY 128
#define MaxZ 256
#define NX (RTM_x - 2 * MaxB)
#define NY (RTM_y - 2 * MaxB)
#define NZ (RTM_z - 2 * MaxB)
#define NUM_INST 3
#define nPE 2
typedef float DATATYPE;

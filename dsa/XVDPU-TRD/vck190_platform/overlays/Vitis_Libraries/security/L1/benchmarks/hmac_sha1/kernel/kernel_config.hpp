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
#ifndef __KERNEL_CONFIG_HPP_
#define __KERNEL_CONFIG_HPP_

// SUB_GRP_SZ * GRP should always be 512
#define SUB_GRP_SZ 16
#define GRP_WIDTH 32
#define GRP_SIZE (GRP_WIDTH / 8)
#define CH_NM SUB_GRP_SZ
#define BURST_LEN 128

#endif

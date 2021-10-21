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

/* This header contains only static config numbers */

/* Hardware related */
#define PU_NM 16         /*!< Number of processing uints */
#define INSTR_DEPTH 4096 /*!< Max number of instructions, the actual one will be given after the RE is pre-compiled */
#define CCLASS_NM 128    /*!< Max number of character classes */
#define CPGP_NM 512      /*!< Max number of capturing groups */
#define MSG_LEN 512      /*!< Max size for each message, in 64-bit */
#define REPEAT_CNT 65536 /*!< Not in use currently */
#define STACK_SIZE 8192  /*!< Max size for the internal stack in hardware regex-VM */

/* Software related */
#define MSG_SZ (MSG_LEN * 8) /*!< Max size for each message, in byte */
#define SLICE_SZ (5242880)   /*!< Max message slice size, in byte */
#define SLICE_NM (256)       /*!< Max section number, a section will never exceed the size of SLICE_MSG_SZ */

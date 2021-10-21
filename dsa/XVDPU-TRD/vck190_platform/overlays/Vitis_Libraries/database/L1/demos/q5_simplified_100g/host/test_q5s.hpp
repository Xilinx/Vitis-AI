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

#if 0
// FIXME override back to 1G size for test
#undef L_MAX_ROW
#define L_MAX_ROW 6001215l
#undef O_MAX_ROW
#define O_MAX_ROW 1500000l
#endif

// by TPCH design, but should be available in decent DBMS implementatin.
#define ORDERKEY_MIN (1l)
#define ORDERKEY_MAX (O_MAX_ROW * 4)

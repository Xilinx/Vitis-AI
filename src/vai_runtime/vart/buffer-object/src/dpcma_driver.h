/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#define BUFFER_OBJECT_IOCTL_MAGIC 'D'
#define BUFFER_OBJECT_SHOW _IOWR(BUFFER_OBJECT_IOCTL_MAGIC, 4, int)
typedef unsigned long phy_addr_t;
struct buffer_object_req_alloc {
  size_t size;
  phy_addr_t phy_addr;
  size_t capacity;
};
#define BUFFER_OBJECT_ALLOC _IOWR(BUFFER_OBJECT_IOCTL_MAGIC, 1, struct buffer_object_req_alloc *)
struct buffer_object_req_free {
  phy_addr_t phy_addr;
  size_t capacity;
};
#define BUFFER_OBJECT_FREE _IOWR(BUFFER_OBJECT_IOCTL_MAGIC, 2, struct buffer_object_req_free *)

#define BUFFER_OBJECT_FROM_CPU_TO_DEVICE (0)
#define BUFFER_OBJECT_FROM_DEVICE_TO_CPU (1)
struct buffer_object_req_sync {
  phy_addr_t phy_addr;
  size_t size;
  int direction;
};
#define BUFFER_OBJECT_SYNC _IOWR(BUFFER_OBJECT_IOCTL_MAGIC, 3, struct buffer_object_req_sync *)

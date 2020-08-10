/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {
std::vector<int> GetDivisors(int number) {
  const int max_divisor = static_cast<int>(std::sqrt(number));
  std::vector<int> divisors;
  // we don't know the number of dividers, so it is just heuristic.
  divisors.reserve(max_divisor / 3 + 1);
  for (int i = 1; i <= max_divisor; ++i) {
    const int d = number / i;
    if (i * d == number) {
      divisors.push_back(i);
      if (d != i) {
        divisors.push_back(d);
      }
    }
  }
  return divisors;
}

std::vector<int> GetDivisorsForRange(int number, int range) {
  const int last_number = number + range;
  const int max_divisor = static_cast<int>(std::sqrt(last_number));
  std::set<int> divisors;
  for (int i = 1; i <= max_divisor; ++i) {
    const int reminder = number % i;
    // iterate through numbers that divisible by i in our range;
    const int first_number = number + (i - reminder) % i;
    if (first_number <= last_number) {
      divisors.insert(i);
    }
    for (int j = first_number; j <= last_number; j += i) {
      const int d = j / i;
      if (d != i) {
        divisors.insert(d);
      }
    }
  }
  return std::vector<int>(divisors.begin(), divisors.end());
}

std::vector<int2> Get2DWorkgroupsEqualTo128() {
  return {{128, 1}, {64, 2}, {32, 4}, {16, 8},
          {8, 16},  {4, 32}, {2, 64}, {1, 128}};
}

std::vector<int> GetPossibleSizes(int number,
                                  WorkGroupSizeAlignment z_alignment) {
  if (z_alignment == WorkGroupSizeAlignment::PRECISE) {
    // we will use for potential sizes, sizes that cover grid preciselly
    // work group size * k (k is integer) == grid_size
    return GetDivisors(number);
  } else {
    // when we chose work group size we can use work group size that
    //   work group size * k (k is integer) != grid_size (slightly bigger)
    // so in this heuristic we trying to find potential size, that satisfies
    //   to this : work group size * k (k is integer) <= grid_size + 5
    //   and this : work group size * k (k is integer) >= grid_size
    return GetDivisorsForRange(number, 5);
  }
}

std::vector<int3> GenerateWorkGroupSizesXY128(
    int3 grid, int max_work_group_size, WorkGroupSizeAlignment z_alignment) {
  std::vector<int3> work_groups;
  work_groups.reserve(32);

  std::vector<int> possible_z_sizes = GetPossibleSizes(grid.z, z_alignment);

  for (int x = 1; x <= max_work_group_size; x *= 2) {
    for (int y = 1; y <= max_work_group_size; y *= 2) {
      int work_group_size_xy = x * y;
      if (work_group_size_xy % 128 != 0 ||
          work_group_size_xy > max_work_group_size) {
        continue;
      }
      for (auto z : possible_z_sizes) {
        if (work_group_size_xy * z > max_work_group_size) {
          continue;
        }
        work_groups.push_back({x, y, z});
      }
    }
  }
  return work_groups;
}

std::vector<int3> GenerateWorkGroupSizesXY128Linear(
    int3 grid, int max_work_group_size, WorkGroupSizeAlignment z_alignment) {
  std::vector<int3> work_groups;
  work_groups.reserve(32);

  std::vector<int> possible_z_sizes = GetPossibleSizes(grid.z, z_alignment);

  for (int x = 128; x <= max_work_group_size && x < grid.x + 128; x += 128) {
    for (auto z : possible_z_sizes) {
      if (x * z <= max_work_group_size) {
        work_groups.push_back({x, 1, z});
      }
    }
  }
  return work_groups;
}

std::vector<int3> GenerateWorkGroupSizes(const int3& grid,
                                         int min_work_group_total_size,
                                         int max_work_group_total_size,
                                         const int3& max_work_group_sizes,
                                         WorkGroupSizeAlignment x_alignment,
                                         WorkGroupSizeAlignment y_alignment,
                                         WorkGroupSizeAlignment z_alignment) {
  std::vector<int3> work_groups;
  work_groups.reserve(64);

  std::vector<int> sizes_x = GetPossibleSizes(grid.x, x_alignment);
  std::vector<int> sizes_y = GetPossibleSizes(grid.y, y_alignment);
  std::vector<int> sizes_z = GetPossibleSizes(grid.z, z_alignment);

  for (auto x : sizes_x) {
    if (x > max_work_group_sizes.x) continue;
    for (auto y : sizes_y) {
      if (y > max_work_group_sizes.y) continue;
      for (auto z : sizes_z) {
        if (z > max_work_group_sizes.z) continue;
        const int work_group_size = x * y * z;
        if (work_group_size < min_work_group_total_size ||
            work_group_size > max_work_group_total_size)
          continue;
        work_groups.push_back({x, y, z});
      }
    }
  }

  return work_groups;
}

void AddCornerCases(const int3& grid, int max_work_group_total_size,
                    const int3& max_work_group_sizes,
                    WorkGroupSizeAlignment x_alignment,
                    WorkGroupSizeAlignment y_alignment,
                    WorkGroupSizeAlignment z_alignment,
                    std::vector<int3>* work_groups) {
  for (int x = 1; x <= 4; ++x) {
    for (int y = 1; y <= 4; ++y) {
      for (int z = 1; z <= 4; ++z) {
        int wg_x = IntegralDivideRoundUp(grid.x, x);
        int wg_y = IntegralDivideRoundUp(grid.y, y);
        int wg_z = IntegralDivideRoundUp(grid.z, z);
        if (wg_x > max_work_group_sizes.x || wg_y > max_work_group_sizes.y ||
            wg_z > max_work_group_sizes.z ||
            wg_x * wg_y * wg_z > max_work_group_total_size) {
          continue;
        }
        if (x_alignment == WorkGroupSizeAlignment::PRECISE &&
            grid.x % wg_x != 0) {
          continue;
        }
        if (y_alignment == WorkGroupSizeAlignment::PRECISE &&
            grid.y % wg_y != 0) {
          continue;
        }
        if (z_alignment == WorkGroupSizeAlignment::PRECISE &&
            grid.z % wg_z != 0) {
          continue;
        }
        work_groups->push_back({wg_x, wg_y, wg_z});
      }
    }
  }

  // this will add at least {1, 1, 1} always.
  for (int x = 1; x <= 4; ++x) {
    for (int y = 1; y <= 4; ++y) {
      for (int z = 1; z <= 4; ++z) {
        if (x > max_work_group_sizes.x || y > max_work_group_sizes.y ||
            z > max_work_group_sizes.z ||
            x * y * z > max_work_group_total_size) {
          continue;
        }
        if (x_alignment == WorkGroupSizeAlignment::PRECISE && grid.x % x != 0) {
          continue;
        }
        if (y_alignment == WorkGroupSizeAlignment::PRECISE && grid.y % y != 0) {
          continue;
        }
        if (z_alignment == WorkGroupSizeAlignment::PRECISE && grid.z % z != 0) {
          continue;
        }
        work_groups->push_back({x, y, z});
      }
    }
  }
}

Status GetBestWorkGroupAlignedToGrid(const TuningParameters& params,
                                     const CLKernel& kernel, const int3& grid,
                                     int3* best_work_group) {
  auto alignment = WorkGroupSizeAlignment::PRECISE;
  std::vector<int3> work_groups = GenerateWorkGroupSizes(
      grid, /*min_work_group_total_size = */ 32, kernel.GetMaxWorkGroupSize(),
      params.info->max_work_group_sizes, alignment, alignment, alignment);
  int best_work_group_index;
  // If the grid parameter too small, method below cannot generate workgroups.
  if (work_groups.empty()) {
    AddCornerCases(grid, kernel.GetMaxWorkGroupSize(),
                   params.info->max_work_group_sizes, alignment, alignment,
                   alignment, &work_groups);
  }
  RETURN_IF_ERROR(params.queue->GetBestWorkGroupIndex(
      kernel, *params.info, grid, work_groups, &best_work_group_index));
  *best_work_group = work_groups[best_work_group_index];
  return OkStatus();
}

int GetPenalty(int grid_size, int group_size) {
  const int reminder = grid_size % group_size;
  return reminder == 0 ? 0 : group_size - reminder;
}

int GetPenalty(int2 grid_size, int2 group_size) {
  const int p_x = GetPenalty(grid_size.x, group_size.x);
  const int p_y = GetPenalty(grid_size.y, group_size.y);
  return p_x * grid_size.y + p_y * grid_size.x + p_x * p_y;
}

int GetMaxSizeWithMinPenalty(int size, int max_size) {
  int best_size = 128;
  int min_penalty = GetPenalty(size, best_size);
  for (int i = 2; i * 128 <= max_size; ++i) {
    if (GetPenalty(size, i * 128) == min_penalty) {
      best_size = i * 128;
    }
  }
  return best_size;
}

int2 GetMaxSizeWithMinPenalty(int2 size, int max_size) {
  std::vector<int2> base_groups = Get2DWorkgroupsEqualTo128();
  int min_penalty = std::numeric_limits<int>::max();
  for (auto group : base_groups) {
    min_penalty = std::min(GetPenalty(size, group), min_penalty);
  }
  for (auto group : base_groups) {
    for (int y = 1; y * group.y <= max_size; ++y) {
      int new_group_y = y * group.y;
      for (int x = 1; x * group.x <= max_size; ++x) {
        int new_group_x = x * group.x;
        if (new_group_x * new_group_y > max_size) {
          break;
        }
        if (GetPenalty(size, int2(new_group_x, new_group_y)) == min_penalty) {
          return int2(new_group_x, new_group_y);
        }
      }
    }
  }
  return int2(0, 0);
}

int GetBiggestDividerWithPriority(int number, int max_divider) {
  if (number % 8 == 0 && 8 <= max_divider) {
    return 8;
  }
  if (number % 4 == 0 && 4 <= max_divider) {
    return 4;
  }
  if (number % 2 == 0 && 2 <= max_divider) {
    return 2;
  }
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

int GetBiggestDivider(int number, int max_divider) {
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

}  // namespace

int3 GetWorkGroupXY128ConvLinear(const int3& grid) {
  int grid_z = GetBiggestDividerWithPriority(grid.z, 4);
  if (grid.x <= 128) {
    return int3(128, 1, grid_z);
  }
  int grid_x = GetMaxSizeWithMinPenalty(grid.x, 512 / grid_z);
  return {grid_x, 1, grid_z};
}

int3 GetWorkGroupXY128Conv(const int3& grid) {
  int grid_z = GetBiggestDividerWithPriority(grid.z, 4);
  if (grid.x <= 16 && grid.y <= 8) {
    return int3(16, 8, grid_z);
  }
  int2 grid_xy = GetMaxSizeWithMinPenalty(int2(grid.x, grid.y), 512 / grid_z);
  return int3(grid_xy.x, grid_xy.y, grid_z);
}

int3 GetWorkGroupXY128Simple(const int3& grid) { return int3(16, 8, 1); }

int3 GetWorkGroup(const int3& grid, int max_size) {
  int wg_z = GetBiggestDividerWithPriority(grid.z, 8);
  int wg_xy_size = max_size / wg_z;
  int wg_x = std::min(IntegralDivideRoundUp(grid.x, 2), wg_xy_size);
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  return int3(wg_x, wg_y, wg_z);
}

int3 GetWorkGroupConv(const int3& grid, int max_size, int max_z_size) {
  int wg_z = GetBiggestDivider(grid.z, max_z_size);
  int wg_xy_size = std::min(256, max_size) / wg_z;
  int wg_x = std::min(grid.x, wg_xy_size);
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  if (wg_y == grid.y && grid.y % 2 == 0) {
    wg_y = grid.y / 2;
  }
  return int3(wg_x, wg_y, wg_z);
}

Status GetBestWorkGroupXY128(const TuningParameters& params,
                             const CLKernel& kernel, const int3& grid,
                             WorkGroupSizeAlignment z_alignment,
                             int3* best_work_group) {
  std::vector<int3> work_groups = GenerateWorkGroupSizesXY128(
      grid, kernel.GetMaxWorkGroupSize(), z_alignment);
  int best_work_group_index;
  RETURN_IF_ERROR(params.queue->GetBestWorkGroupIndex(
      kernel, *params.info, grid, work_groups, &best_work_group_index));
  *best_work_group = work_groups[best_work_group_index];
  return OkStatus();
}

Status GetBestWorkGroupXY128Linear(const TuningParameters& params,
                                   const CLKernel& kernel, const int3& grid,
                                   WorkGroupSizeAlignment z_alignment,
                                   int3* best_work_group) {
  std::vector<int3> work_groups = GenerateWorkGroupSizesXY128Linear(
      grid, kernel.GetMaxWorkGroupSize(), z_alignment);
  int best_work_group_index;
  RETURN_IF_ERROR(params.queue->GetBestWorkGroupIndex(
      kernel, *params.info, grid, work_groups, &best_work_group_index));
  *best_work_group = work_groups[best_work_group_index];
  return OkStatus();
}

bool XY128RequiresMoreWorkGroupsThenXY128Linear(int width, int height) {
  int planar_work_groups = IntegralDivideRoundUp(width * height, 128);
  auto base_work_groups = Get2DWorkgroupsEqualTo128();
  bool have_equal_work_groups = false;
  for (auto& work_group : base_work_groups) {
    int x_groups = IntegralDivideRoundUp(width, work_group.x);
    int y_groups = IntegralDivideRoundUp(height, work_group.y);
    int xy_groups = x_groups * y_groups;
    if (xy_groups == planar_work_groups) {
      have_equal_work_groups = true;
      break;
    }
  }
  return !have_equal_work_groups;
}

Status GetBestWorkGroup(const TuningParameters& params, const CLKernel& kernel,
                        const int3& grid, int3* best_work_group) {
  switch (params.tuning_type) {
    case TuningType::FAST:
      if (params.info->vendor != Vendor::QUALCOMM) {
        *best_work_group = int3(8, 4, 1);
        return OkStatus();
      } else {
        *best_work_group = GetWorkGroup(grid, kernel.GetMaxWorkGroupSize());
        return OkStatus();
      }
    case TuningType::EXHAUSTIVE:
      return GetBestWorkGroupAlignedToGrid(params, kernel, grid,
                                           best_work_group);
    default:
      *best_work_group = {8, 4, 1};
      return OkStatus();
  }
}

Status GetBestWorkGroupConv(const TuningParameters& params,
                            const CLKernel& kernel, const int3& grid,
                            int3* best_work_group) {
  switch (params.tuning_type) {
    case TuningType::FAST:
      if (params.info->vendor != Vendor::QUALCOMM) {
        *best_work_group = int3(8, 4, 1);
        return OkStatus();
      } else {
        int max_z_size = params.info->adreno_info.gpu_version < 400 ? 16 : 64;
        *best_work_group =
            GetWorkGroupConv(grid, kernel.GetMaxWorkGroupSize(), max_z_size);
        return OkStatus();
      }
    case TuningType::EXHAUSTIVE:
      return GetBestWorkGroupAlignedToGrid(params, kernel, grid,
                                           best_work_group);
    default:
      *best_work_group = {8, 4, 1};
      return OkStatus();
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

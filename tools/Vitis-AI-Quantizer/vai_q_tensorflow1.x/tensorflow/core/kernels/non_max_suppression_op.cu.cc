/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <limits>

#include "absl/strings/str_cat.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/non_max_suppression_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/stream_executor/stream_executor.h"

#define TF_RETURN_IF_CUDA_ERROR(result)                   \
  do {                                                    \
    cudaError_t error(result);                            \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {         \
      return errors::Internal("Cuda call failed with ",   \
                              cudaGetErrorString(error)); \
    }                                                     \
  } while (0)

#define TF_OP_REQUIRES_CUDA_SUCCESS(context, result)                   \
  do {                                                                 \
    cudaError_t error(result);                                         \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {                      \
      context->SetStatus(errors::Internal("Cuda call failed with",     \
                                          cudaGetErrorString(error))); \
      return;                                                          \
    }                                                                  \
  } while (0)

struct __align__(16) Box {
  float x1, y1, x2, y2;
};

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

// This is the width of the bitmask for masking boxes for each thread.
// This needs to be a multiple of 2(a POD width usually) so that division and
// modulo can be implemented as bit operations during host selection.
constexpr int kNmsBoxesPerThread = 8 * sizeof(int);
// Helper to calculate modulo mask and shift bits.
// For kNmsBoxesPerThread=32 ModuloMask will be 31, i.e 0x1F thus
// i % 32 == i & 31. Similarly ShiftBits will be 5 so that
// i / 32 == i >> 5. Using these bit operations should reduce the stall on host
// thread.
constexpr int NumBits(int n) { return (n == 0) ? 0 : NumBits(n >> 1) + 1; }
constexpr int kNmsBoxesPerThreadModuloMask = kNmsBoxesPerThread - 1;
constexpr int kNmsBoxesPerThreadShiftBits =
    NumBits(kNmsBoxesPerThreadModuloMask);

constexpr int kNmsBlockDim = 16;
constexpr int kNmsBlockDimMax = 128;
constexpr int kNmsChunkSize = 2000;

template <typename T>
__device__ EIGEN_STRONG_INLINE void Swap(T& a, T& b) {
  T c(a);
  a = b;
  b = c;
}

// Check whether two boxes have an IoU greater than threshold.
template <typename T>
__device__ EIGEN_STRONG_INLINE bool OverThreshold(const Box* a, const Box* b,
                                                  const float a_area,
                                                  const T iou_threshold) {
  const float b_area = (b->x2 - b->x1) * (b->y2 - b->y1);
  if (a_area == 0.0f || b_area == 0.0f) return false;
  const float xx1 = fmaxf(a->x1, b->x1);
  const float yy1 = fmaxf(a->y1, b->y1);
  const float xx2 = fminf(a->x2, b->x2);
  const float yy2 = fminf(a->y2, b->y2);

  // fdimf computes the positive difference between xx2+1 and xx1.
  const float w = fdimf(xx2, xx1);
  const float h = fdimf(yy2, yy1);
  const float intersection = w * h;

  // Testing for aa/bb > t
  // eq with aa > bb*t (b is !=0)
  // avoiding divisions.
  const float aa = intersection;
  const float bb = a_area + b_area - intersection;
  const float bt = bb * iou_threshold;
  return aa > bt;
}

template <bool flip_box>
__device__ EIGEN_STRONG_INLINE void Flipped(Box& box);

template <>
__device__ EIGEN_STRONG_INLINE void Flipped<false>(Box& box) {}

template <>
__device__ EIGEN_STRONG_INLINE void Flipped<true>(Box& box) {
  if (box.x1 > box.x2) Swap(box.x1, box.x2);
  if (box.y1 > box.y2) Swap(box.y1, box.y2);
}
template <typename T>
__device__ EIGEN_STRONG_INLINE bool CheckBit(T* bit_mask, int bit) {
  constexpr int kShiftLen = NumBits(8 * sizeof(T)) - 1;
  constexpr int kRemainderMask = 8 * sizeof(T) - 1;
  int bin = bit >> kShiftLen;
  return (bit_mask[bin] >> (bit & kRemainderMask)) & 1;
}

// Produce a global bitmask (result_mask) of selected boxes from bitmask
// generated by NMSKernel Abort early if max_boxes boxes are selected. Bitmask
// is num_boxes*bit_mask_len bits indicating whether to keep or remove a box.
__global__ void NMSReduce(const int* bitmask, const int bit_mask_len,
                          const int num_boxes, const int max_boxes,
                          char* result_mask) {
  extern __shared__ int local[];
  // set global mask to accept all boxes
  for (int box : CudaGridRangeX(bit_mask_len)) {
    local[box] = 0xFFFFFFFF;
  }
  __syncthreads();
  int accepted_boxes = 0;
  for (int box = 0; box < num_boxes - 1; ++box) {
    // if current box is masked by an earlier box, skip it.
    if (!CheckBit(local, box)) {
      continue;
    }
    accepted_boxes += 1;
    int offset = box * bit_mask_len;
    // update global mask with current box's mask
    for (int b : CudaGridRangeX(bit_mask_len)) {
      local[b] &= ~bitmask[offset + b];
    }
    __syncthreads();
    if (accepted_boxes > max_boxes) break;
  }
  // copy global mask to result_max char array. char array is needed for
  // cub::DeviceSelect later.
  for (int box : CudaGridRangeX(num_boxes)) {
    result_mask[box] = CheckBit(local, box);
  }
}

// For each box, compute a bitmask of boxes which has an overlap with given box
// above threshold.
//
// Starting from highes scoring box, mark any box which has IoU>threshold with
// given box. Each thread processes a kNmsBoxesPerThread boxes per stride, and
// each box has bitmask of overlaps of length bit_mask_len.
//
// If flip_box is true boxes may have x1>x2 and or y1>y2. If so change the
// coordinates such that for all boxes x1<x2 and y1<y2. Else boxes should have
// x1<x2 and y1<y2.
template <bool flip_box>
__launch_bounds__(kNmsBlockDim* kNmsBlockDim, 4) __global__
    void NMSKernel(const Box* d_desc_sorted_boxes, const int num_boxes,
                   const float iou_threshold, const int bit_mask_len,
                   int* d_delete_mask) {
  // Storing boxes used by this CUDA block in the shared memory.
  __shared__ Box shared_i_boxes[kNmsBlockDim];
  // Same thing with areas
  __shared__ float shared_i_areas[kNmsBlockDim];
  // The condition of the for loop is common to all threads in the block.
  // This is necessary to be able to call __syncthreads() inside of the loop.
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < num_boxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i = i_block_offset + threadIdx.x;
    if (i < num_boxes) {
      // One 1D line load the boxes for x-dimension.
      if (threadIdx.y == 0) {
        Box box = d_desc_sorted_boxes[i];
        Flipped<flip_box>(box);
        shared_i_boxes[threadIdx.x] = box;
        shared_i_areas[threadIdx.x] = (box.x2 - box.x1) * (box.y2 - box.y1);
      }
    }
    __syncthreads();
    for (int j_thread_offset =
             kNmsBoxesPerThread * (blockIdx.y * blockDim.y + threadIdx.y);
         j_thread_offset < num_boxes;
         j_thread_offset += kNmsBoxesPerThread * blockDim.y * gridDim.y) {
      // Note : We can do everything using multiplication,
      // and use fp16 - we are comparing against a low precision
      // threshold.
      int above_threshold = 0;
      // Make sure that threads are within valid domain.
      bool valid = false;
      // Loop over the next kNmsBoxesPerThread boxes and set corresponding bit
      // if it is overlapping with current box
      for (int ib = 0; ib < kNmsBoxesPerThread; ++ib) {
        // This thread will compare Box i and Box j.
        const int j = j_thread_offset + ib;
        if (i >= j || i >= num_boxes || j >= num_boxes) continue;
        valid = true;
        Box j_box = d_desc_sorted_boxes[j];
        const Box i_box = shared_i_boxes[threadIdx.x];
        Flipped<flip_box>(j_box);
        if (OverThreshold<float>(&i_box, &j_box, shared_i_areas[threadIdx.x],
                                 iou_threshold)) {
          // we have score[j] <= score[i].
          above_threshold |= (1U << ib);
        }
      }
      if (valid) {
        d_delete_mask[i * bit_mask_len + j_thread_offset / kNmsBoxesPerThread] =
            above_threshold;
      }
    }
    __syncthreads();  // making sure everyone is done reading shared memory.
  }
}
// Variadic template helpers for Index selecting multiple arrays at the same
// time
template <typename Index>
__device__ EIGEN_STRONG_INLINE void SelectHelper(const Index i_selected,
                                                 const Index i_original) {}

template <typename Index, typename T, typename... Args>
__device__ EIGEN_STRONG_INLINE void SelectHelper(const Index i_selected,
                                                 const Index i_original,
                                                 const T* original, T* selected,
                                                 Args... args) {
  selected[i_selected] = original[i_original];
  SelectHelper(i_selected, i_original, args...);
}

// Helper template to select elements from original arrays using the index
// mapping and store into selected array. Each array sharing same mapping need
// to be passed as pairs of pointers to original and selected arrays. For
// selecting 2 arrays call would be
// IndexMultiSelect(num_elements, indices, original1 ,selected1, original2,
// selected2).
template <typename Index, typename T, typename... Args>
__global__ void IndexMultiSelect(const int num_elements, const Index* indices,
                                 const T* original, T* selected, Args... args) {
  for (const int idx : CudaGridRangeX(num_elements)) {
    SelectHelper(idx, indices[idx], original, selected, args...);
  }
}

template <typename T>
__global__ void Iota(const int num_elements, const T offset, T* to_fill) {
  for (int idx : CudaGridRangeX(num_elements)) {
    to_fill[idx] = static_cast<T>(idx) + offset;
  }
}

Status NmsGpu(const float* d_sorted_boxes_float_ptr, const int num_boxes,
              const float iou_threshold, int* d_selected_indices, int* h_nkeep,
              OpKernelContext* context, const int max_boxes, bool flip_boxes) {
  // Making sure we respect the __align(16)__
  // we promised to the compiler.
  auto iptr = reinterpret_cast<std::uintptr_t>(d_sorted_boxes_float_ptr);
  if ((iptr & 15) != 0) {
    return errors::InvalidArgument("Boxes should be aligned to 16 Bytes.");
  }
  // allocate bitmask arrays on host and on device
  Tensor h_num_selected, d_nms_mask;
  const int bit_mask_len =
      (num_boxes + kNmsBoxesPerThread - 1) / kNmsBoxesPerThread;

  int64 max_nms_mask_size = num_boxes * bit_mask_len;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({max_nms_mask_size}), &d_nms_mask));
  // reset data sensitive tensors
  auto device = context->eigen_gpu_device();
  auto config = GetGpuLaunchConfig(d_nms_mask.NumElements(), device);
  TF_CHECK_OK(GpuLaunchKernel(SetZero<int>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count,
                              d_nms_mask.flat<int32>().data()));

  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  // Size of this buffer can be reduced to kNmsChunkSize*bit_mask_len*2 and
  // using it as a ring buffer. However savings should be a few MB .
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({1}), &h_num_selected, alloc_attr));

  int* d_delete_mask = d_nms_mask.flat<int>().data();
  int* h_selected_count = h_num_selected.flat<int>().data();
  const Box* d_sorted_boxes =
      reinterpret_cast<const Box*>(d_sorted_boxes_float_ptr);
  dim3 block_dim, thread_block;
  int num_blocks = (num_boxes + kNmsBlockDim - 1) / kNmsBlockDim;
  num_blocks = std::max(std::min(num_blocks, kNmsBlockDimMax), 1);
  block_dim.x = num_blocks;
  block_dim.y = num_blocks;
  block_dim.z = 1;
  thread_block.x = kNmsBlockDim;
  thread_block.y = kNmsBlockDim;
  thread_block.z = 1;
  if (flip_boxes) {
    TF_CHECK_OK(GpuLaunchKernel(NMSKernel<true>, block_dim, thread_block, 0,
                                device.stream(), d_sorted_boxes, num_boxes,
                                iou_threshold, bit_mask_len, d_delete_mask));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(NMSKernel<false>, block_dim, thread_block, 0,
                                device.stream(), d_sorted_boxes, num_boxes,
                                iou_threshold, bit_mask_len, d_delete_mask));
  }
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  // Overlapping CPU computes and D2H memcpy
  // both take about the same time

  config = GetGpuLaunchConfig(num_boxes, device);
  Tensor selected_boxes;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({num_boxes}), &selected_boxes));
  Tensor d_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_indices));
  TF_CHECK_OK(GpuLaunchKernel(Iota<int>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count, 0,
                              d_indices.flat<int>().data()));

  char* selected = (char*)(selected_boxes.flat<int8>().data());
  TF_CHECK_OK(GpuLaunchKernel(NMSReduce, 1, 1024, bit_mask_len * sizeof(int),
                              device.stream(), d_delete_mask, bit_mask_len,
                              num_boxes, max_boxes, selected));
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  // do Cub::deviceSelect::flagged
  size_t flagged_buffer_size = 0;
  cub::DeviceSelect::Flagged(static_cast<void*>(nullptr),  // temp_storage
                             flagged_buffer_size,
                             static_cast<int*>(nullptr),   // input
                             static_cast<char*>(nullptr),  // selection flag
                             static_cast<int*>(nullptr),   // selected items
                             static_cast<int*>(nullptr),   // num_selected
                             num_boxes, device.stream());
  Tensor cub_scratch;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)flagged_buffer_size}),
      &cub_scratch));
  Tensor d_num_selected;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({1}), &d_num_selected));

  cub::DeviceSelect::Flagged(
      (void*)cub_scratch.flat<int8>().data(),  // temp_storage
      flagged_buffer_size,
      d_indices.flat<int>().data(),  // input
      selected,                      // selection flag
      d_selected_indices,            // selected items
      d_num_selected.flat<int>().data(), num_boxes, device.stream());
  cudaEvent_t copy_done;
  TF_RETURN_IF_CUDA_ERROR(
      cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
  device.memcpyDeviceToHost(h_selected_count, d_num_selected.flat<int>().data(),
                            sizeof(int));
  TF_RETURN_IF_CUDA_ERROR(cudaEventRecord(copy_done, device.stream()));
  TF_RETURN_IF_CUDA_ERROR(cudaEventSynchronize(copy_done));
  *h_nkeep = *h_selected_count;
  cudaEventDestroy(copy_done);
  return Status::OK();
}

struct GreaterThanCubOp {
  float threshold_;
  __host__ __device__ __forceinline__ GreaterThanCubOp(float threshold)
      : threshold_(threshold) {}
  __host__ __device__ __forceinline__ bool operator()(const float& val) const {
    return (val > threshold_);
  }
};
// Use DeviceSelect::If to count number of elements.
// TODO(sami) Not really a good way. Perhaps consider using thrust?
template <typename Op>
Status CountIf(OpKernelContext* context, const float* dev_array, const Op& op,
               int num_elements, int* result) {
  Tensor scratch_output;
  Tensor workspace;
  Tensor element_count;
  size_t workspace_size = 0;
  auto cuda_stream = tensorflow::GetGpuStream(context);
  auto device = context->eigen_gpu_device();
  cub::DeviceSelect::If(nullptr, workspace_size, static_cast<float*>(nullptr),
                        static_cast<float*>(nullptr),
                        static_cast<int*>(nullptr), num_elements, op);

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_elements}), &scratch_output));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)workspace_size}), &workspace));
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({1}), &element_count));
  cudaEvent_t copy_done;
  TF_RETURN_IF_CUDA_ERROR(
      cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
  TF_RETURN_IF_CUDA_ERROR(cub::DeviceSelect::If(
      workspace.flat<int8>().data(), workspace_size, dev_array,
      scratch_output.flat<float>().data(), element_count.flat<int32>().data(),
      num_elements, op, cuda_stream));
  device.memcpyDeviceToHost(result, element_count.flat<int32>().data(),
                            sizeof(int));
  TF_RETURN_IF_CUDA_ERROR(cudaEventRecord(copy_done, device.stream()));
  TF_RETURN_IF_CUDA_ERROR(cudaEventSynchronize(copy_done));
  return Status::OK();
}

Status DoNMS(OpKernelContext* context, const Tensor& boxes,
             const Tensor& scores, const int64_t max_output_size,
             const float iou_threshold_val, const float score_threshold) {
  const int output_size = max_output_size;
  int num_boxes = boxes.dim_size(0);
  size_t cub_sort_temp_storage_bytes = 0;
  auto cuda_stream = GetGpuStream(context);
  auto device = context->eigen_gpu_device();
  // Calling cub with nullptrs as inputs will make it return
  // workspace size needed for the operation instead of doing the operation.
  // In this specific instance, cub_sort_temp_storage_bytes will contain the
  // necessary workspace size for sorting after the call.
  if (num_boxes == 0) {
    Tensor* output_indices = nullptr;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, TensorShape({0}), &output_indices));
    return Status::OK();
  }

  cudaError_t cuda_ret = cub::DeviceRadixSort::SortPairsDescending(
      nullptr, cub_sort_temp_storage_bytes,
      static_cast<float*>(nullptr),  // scores
      static_cast<float*>(nullptr),  // sorted scores
      static_cast<int*>(nullptr),    // input indices
      static_cast<int*>(nullptr),    // sorted indices
      num_boxes,                     // num items
      0, 8 * sizeof(float),          // sort all bits
      cuda_stream);
  TF_RETURN_IF_CUDA_ERROR(cuda_ret);
  Tensor d_cub_sort_buffer;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_sort_temp_storage_bytes}),
      &d_cub_sort_buffer));
  Tensor d_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_indices));
  Tensor d_sorted_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_sorted_indices));
  Tensor d_selected_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_selected_indices));
  Tensor d_sorted_scores;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_boxes}), &d_sorted_scores));
  Tensor d_sorted_boxes;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_boxes, 4}), &d_sorted_boxes));

  // this will return sorted scores and their indices
  auto config = GetGpuLaunchConfig(num_boxes, device);
  // initialize box and score indices
  TF_CHECK_OK(GpuLaunchKernel(Iota<int>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count, 0,
                              d_indices.flat<int>().data()));
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  cuda_ret = cub::DeviceRadixSort::SortPairsDescending(
      d_cub_sort_buffer.flat<int8>().data(), cub_sort_temp_storage_bytes,
      scores.flat<float>().data(), d_sorted_scores.flat<float>().data(),
      d_indices.flat<int>().data(), d_sorted_indices.flat<int>().data(),
      num_boxes, 0,
      8 * sizeof(float),  // sort all bits
      cuda_stream);
  TF_RETURN_IF_CUDA_ERROR(cuda_ret);

  // get pointers for easy access
  const float4* original_boxes =
      reinterpret_cast<const float4*>(boxes.flat<float>().data());
  float4* sorted_boxes =
      reinterpret_cast<float4*>(d_sorted_boxes.flat<float>().data());
  const int* sorted_indices = d_sorted_indices.flat<int>().data();
  // sort boxes using indices
  TF_CHECK_OK(GpuLaunchKernel(IndexMultiSelect<int, float4>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count, sorted_indices,
                              original_boxes, sorted_boxes));
  int limited_num_boxes = num_boxes;
  // filter boxes by scores if nms v3
  if (score_threshold > std::numeric_limits<float>::lowest()) {
    GreaterThanCubOp score_limit(score_threshold);
    TF_RETURN_IF_ERROR(CountIf(context, d_sorted_scores.flat<float>().data(),
                               score_limit, num_boxes, &limited_num_boxes));
    if (limited_num_boxes == 0) {
      Tensor* output_indices = nullptr;
      VLOG(1) << "Number of boxes above score threshold " << score_threshold
              << " is 0";
      TF_RETURN_IF_ERROR(
          context->allocate_output(0, TensorShape({0}), &output_indices));
      return Status::OK();
    } else {
      VLOG(2) << "Number of boxes above threshold=" << score_threshold << " is "
              << limited_num_boxes;
    }
  }
  int num_to_keep = 0;
  // There is no guarantee that boxes are given in the for x1<x2 and/or y1<y2,
  // flip boxes if necessary!
  const bool flip_boxes = true;
  auto status = NmsGpu(d_sorted_boxes.flat<float>().data(), limited_num_boxes,
                       iou_threshold_val, d_selected_indices.flat<int>().data(),
                       &num_to_keep, context, output_size, flip_boxes);
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  if (!status.ok()) {
    context->SetStatus(status);
    return status;
  }
  Tensor* output_indices = nullptr;
  int num_outputs = std::min(num_to_keep, output_size);  // no padding!
  TF_RETURN_IF_ERROR(
      context->allocate_output(0, TensorShape({num_outputs}), &output_indices));
  if (num_outputs == 0) return Status::OK();
  config = GetGpuLaunchConfig(num_outputs, device);
  TF_CHECK_OK(GpuLaunchKernel(
      IndexMultiSelect<int, int>, config.block_count, config.thread_per_block,
      0, device.stream(), config.virtual_thread_count,
      d_selected_indices.flat<int>().data(), sorted_indices,
      (*output_indices).flat<int>().data()));
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  return Status::OK();
}

class NonMaxSuppressionV2GPUOp : public OpKernel {
 public:
  explicit NonMaxSuppressionV2GPUOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    OP_REQUIRES(context, boxes.dims() == 2,
                errors::InvalidArgument("boxes must be a rank 2 tensor!"));
    int num_boxes = boxes.dim_size(0);
    OP_REQUIRES(context, boxes.dim_size(1) == 4,
                errors::InvalidArgument("boxes must be Nx4"));
    OP_REQUIRES(context, scores.dims() == 1,
                errors::InvalidArgument("scores must be a vector!"));
    OP_REQUIRES(
        context, scores.dim_size(0) == num_boxes,
        errors::InvalidArgument(
            "scores has incompatible shape"));  // message must be exactly this
                                                // otherwise tests fail!
    if (num_boxes == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }
    const int64_t output_size = max_output_size.scalar<int>()();
    OP_REQUIRES_OK(
        context,
        DoNMS(context, boxes, scores, output_size, iou_threshold_val,
              /*score_threshold is float min if score threshold is disabled*/
              std::numeric_limits<float>::lowest()));
  }
};

class NonMaxSuppressionV3GPUOp : public OpKernel {
 public:
  explicit NonMaxSuppressionV3GPUOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    OP_REQUIRES(context, boxes.dims() == 2,
                errors::InvalidArgument("boxes must be a rank 2 tensor!"));
    int num_boxes = boxes.dim_size(0);
    OP_REQUIRES(context, boxes.dim_size(1) == 4,
                errors::InvalidArgument("boxes must be Nx4"));
    OP_REQUIRES(context, scores.dims() == 1,
                errors::InvalidArgument("scores must be a vector!"));
    OP_REQUIRES(
        context, scores.dim_size(0) == num_boxes,
        errors::InvalidArgument(
            "scores has incompatible shape"));  // message must be exactly this
                                                // otherwise tests fail!
    if (num_boxes == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }
    const int output_size = max_output_size.scalar<int>()();
    OP_REQUIRES_OK(context, DoNMS(context, boxes, scores, output_size,
                                  iou_threshold_val, score_threshold_val));
  }
};

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV2")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size"),
                        NonMaxSuppressionV2GPUOp);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV3")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size")
                            .HostMemory("score_threshold"),
                        NonMaxSuppressionV3GPUOp);

}  // namespace tensorflow
#endif

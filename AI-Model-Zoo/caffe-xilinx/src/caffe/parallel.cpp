#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
// #include "boost/thread/latch.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/gpu_memory.hpp"
#ifdef USE_NCCL
#include "caffe/util/nccl.hpp"
#endif

namespace caffe {

shared_ptr<boost::barrier> bar;

enum Op {
  copy,
  copy_mask,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff,
  replace_cpu_mask,
  replace_gpu_mask
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        ptr += size;
        break;
      }
      case copy_mask: {
        // Init buffer to current values of blobs
        if (blobs[i]->enable_mask()) {
          caffe_copy(size,
                     reinterpret_cast<const Dtype*>(blobs[i]->mask()->cpu_data()),
                     ptr);
          ptr += size;
        }
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        ptr += size;
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        ptr += size;
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        ptr += size;
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        ptr += size;
        break;
      case replace_cpu_mask:
        if (blobs[i]->enable_mask()) {
          blobs[i]->mask()->set_cpu_data(ptr);
          ptr += size;
        }
        break;
      case replace_gpu_mask:
        if (blobs[i]->enable_mask()) {
          blobs[i]->mask()->set_gpu_data(ptr);
          ptr += size;
        }
        break;
    }
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

// Buffer size necessary to store blobs mask
template<typename Dtype>
static size_t total_mask_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i) {
    if ( params[i]->enable_mask() ){
      size += params[i]->count();
    }
  }
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      mask_size_(total_mask_size<Dtype>(root_solver->net()->learnable_params())),
      data_(NULL),
      diff_(NULL),
      mask_(NULL) {
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  buffer_device_ = device;
  stream_ = GPUMemory::device_stream(buffer_device_);
  GPUMemory::allocate(reinterpret_cast<void **>(&data_),
      size_ * sizeof(Dtype), device, stream_);

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  GPUMemory::allocate(reinterpret_cast<void **>(&diff_),
      size_ * sizeof(Dtype), device, stream_);
  caffe_gpu_set(size_, Dtype(0), diff_);

  // copy mask
  GPUMemory::allocate(reinterpret_cast<void **>(&mask_),
      mask_size_ * sizeof(Dtype), device, stream_);
  apply_buffers(net, mask_, mask_size_, copy_mask);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
  GPUMemory::deallocate(data_, buffer_device_, stream_);
  GPUMemory::deallocate(diff_, buffer_device_, stream_);
  GPUMemory::deallocate(mask_, buffer_device_, stream_);
#endif
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
  apply_buffers(net, mask_, mask_size_, replace_gpu_mask);
}

void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  pairs->push_back(DevicePair(-1, devices[0]));
  for (int i = 0; i < devices.size() - 1; ++i) {
    pairs->push_back(DevicePair(devices[i], devices[i + 1]));
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        int rank, int nranks, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      rank_(rank),
      nranks_(nranks),
      parent_(),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_(),
      params_(param),
      per_parameter_reduce_(param.per_parameter_reduce()) {
#ifndef USE_NCCL
  LOG(FATAL) << "USE_NCCL := 1 must be specified for multi-GPU";
#endif

#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (rank == 0) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(caffe::SolverRegistry<Dtype>::CreateSolver(param,
        root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

#if defined(USE_NCCL)
  nccl_comms_.resize(1);
#endif
  comm_streams_.resize(1);
  CUDA_CHECK(cudaStreamCreateWithFlags(&comm_streams_[0],
                                       cudaStreamNonBlocking));

  CHECK_GT(comm_streams_.size(), 0);
  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

#ifndef CPU_ONLY
#ifdef USE_NCCL
template<typename Dtype>
void P2PSync<Dtype>::setNCCLComm(ncclComm_t comm) {
  this->nccl_comms_[0] = comm;
}

template<typename Dtype>
ncclComm_t P2PSync<Dtype>::getNCCLComm() {
  return this->nccl_comms_[0];
}
#endif

template<typename Dtype>
cudaStream_t P2PSync<Dtype>::getCommStream() {
  return this->comm_streams_[0];
}
#endif

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  for (int i = 0; i < comm_streams_.size(); ++i) {
    cudaStreamDestroy(comm_streams_[i]);
  }

#ifdef USE_NCCL
  for (int i = 0; i < nccl_comms_.size(); ++i) {
    ncclCommDestroy(nccl_comms_[i]);
  }
#endif  // USE_NCCL

#endif
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  // solver_->Step(solver_->param().max_iter() - initial_iter_);
  solver_->Solve();
}

template<typename Dtype>
void P2PSync<Dtype>::soft_barrier() {
#ifndef CPU_ONLY
  // CPU barrier to avoid busy-polling on the GPU.
  bar->wait();
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::sync_data() {
#ifndef CPU_ONLY
#ifdef USE_NCCL
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  NCCL_CHECK(ncclBcast(data_, size_, nccl::dataType<Dtype>::type, 0,
      getNCCLComm(), getCommStream()));
  CUDA_CHECK(cudaStreamSynchronize(getCommStream()));
#endif  // USE_NCCL
  const vector<Blob<Dtype>*>& net =
      solver_->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef USE_NCCL
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  NCCL_CHECK(ncclBcast(data_, size_, nccl::dataType<Dtype>::type, 0,
      getNCCLComm(), getCommStream()));
  CUDA_CHECK(cudaStreamSynchronize(getCommStream()));
#endif  // USE_NCCL
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::allreduce() {
#ifndef CPU_ONLY
#ifdef USE_NCCL
  // only reduce if we haven't in the bwd pass
  if (!per_parameter_reduce_) {
    bar->wait();
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    NCCL_CHECK(ncclAllReduce(diff_, diff_, size_, nccl::dataType<Dtype>::type,
        ncclSum, getNCCLComm(), getCommStream()));
    caffe_gpu_scal(size_, (Dtype)1.0 / Caffe::solver_count(), diff_,
        getCommStream());
  }
#endif  // USE_NCCL
#endif  // CPU_ONLY
}

template<typename Dtype>
void P2PSync<Dtype>::allreduce(int param_id) {
#ifndef CPU_ONLY
#ifdef USE_NCCL
  // reduce aynchronously in the bwd path
  if (per_parameter_reduce_) {
    bar->wait();
    const vector<shared_ptr<Blob<Dtype> > >& params = solver_->net()->params();
    NCCL_CHECK(ncclAllReduce(params[param_id]->gpu_diff(),
                             params[param_id]->mutable_gpu_diff(),
                             params[param_id]->count(),
                             nccl::dataType<Dtype>::type,
                             ncclSum,
                             getNCCLComm(),
                             getCommStream()));
    caffe_gpu_scal(params[param_id]->count(), (Dtype)1. / Caffe::solver_count(),
        params[param_id]->mutable_gpu_diff(), getCommStream());
  }
#endif  // USE_NCCL
#endif  // CPU_ONLY
}

template <typename Dtype>
void P2PSync<Dtype>::syncCommStream() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamSynchronize(comm_streams_[0]));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  bar.reset(new boost::barrier(gpus.size()));
  SolverParameter param = solver_->param();
  this->shared_ = make_shared<SharedScores<float>>(nranks_);
  for (int i = 1; i < gpus.size(); ++i) {
    param.set_device_id(gpus[i]);
    syncs[i].reset(new P2PSync<Dtype>(solver_, i, gpus.size(), param));
    syncs[i]->shared_ = this->shared_;
  }
#ifdef USE_NCCL
  ncclComm_t *comms = new ncclComm_t[nranks_];
  int *gpu_list = new int[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    gpu_list[i] = gpus[i];
  }
  NCCL_CHECK(ncclCommInitAll(comms, nranks_, gpu_list));

  this->setNCCLComm(comms[0]);

  for (int i = 1; i < nranks_; ++i) {
    syncs[i]->setNCCLComm(comms[i]);
  }
  delete[] comms;
  delete[] gpu_list;
#else
  LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif  // USE_NCCL

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  if (this->solver_->Solve()) {
    for (int i = 1; i < syncs.size(); ++i) {
      syncs[i]->solver_->request_early_exit();
    }
  }

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

// master thread gets aggregate of results for output
template <typename Dtype>
void P2PSync<Dtype>::aggregateTestResults(float* loss, vector<float>* scores) {
  // only run on master thread
  if (this->rank_ == 0) {
    // initialize results
    *loss = 0.F;
    for (size_t i = 0; i < scores->size(); ++i) {
      (*scores)[i] = 0.F;
    }
    // all test threads
    for (size_t i = 0; i < nranks_; ++i) {
      vector<float>& shared_scr = shared_->rank_scores(i);
      *loss += shared_scr[0];
      // all scores within each test thread
      for (size_t j = 0; j < scores->size(); ++j) {
        (*scores)[j] += shared_scr[j+1];
      }
    }
  }
}

template <typename Dtype>
void P2PSync<Dtype>::saveTestResults(float loss, const vector<float>& scores) {
  vector<float>& shared_scr = shared_->rank_scores(this->rank_);
  shared_scr[0] = loss;
  for (size_t i = 0; i < scores.size(); ++i) {
    shared_scr[i+1] = scores[i];
  }
}

uint32_t batch_per_gpu(uint32_t total) {
  int solver_count = Caffe::solver_count();
  if (total == 0 || total % solver_count != 0) {
    uint32_t new_total = total + (solver_count - (total % solver_count));
    LOG(WARNING) << "Batch size must be divisible by the number of solvers (GPUs): "
        << "it's been adjusted from " << total << " to " << new_total;
    total = new_total;
  }
  return total / solver_count;
}

template<typename Dtype>
void P2PSync<Dtype>::divide_batch_size(NetParameter* net) {
  for (int i = 0; i < net->layer_size(); ++i) {
    if (net->layer(i).has_data_param()) {
      if (net->layer(i).data_param().has_batch_size()) {
        uint32_t total = net->layer(i).data_param().batch_size();
        if (net->state().phase() == TEST &&
            !net->layer(i).data_param().multi_gpu_testing()) {
          LOG(INFO) << "Testing with one gpu, batch size kept " << total;
          net->mutable_layer(i)->mutable_data_param()->set_batch_size(total);
        } else {
          uint32_t batch = batch_per_gpu(total);
          net->mutable_layer(i)->mutable_data_param()->set_batch_size(batch);
        }
      }
    }
    if (net->layer(i).has_hdf5_data_param()) {
      if (net->layer(i).hdf5_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).hdf5_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_hdf5_data_param()->set_batch_size(batch);
      }
    }
    if (net->layer(i).has_image_data_param()) {
      if (net->layer(i).image_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).image_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_image_data_param()->set_batch_size(
            batch);
      }
    }
    if (net->layer(i).has_memory_data_param()) {
      if (net->layer(i).memory_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).memory_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_memory_data_param()->set_batch_size(
            batch);
      }
    }
    if (net->layer(i).has_window_data_param()) {
      if (net->layer(i).window_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).window_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_window_data_param()->set_batch_size(
            batch);
      }
    }
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);

}  // namespace caffe

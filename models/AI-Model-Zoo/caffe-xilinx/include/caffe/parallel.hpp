#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

// #include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

#ifdef USE_NCCL
#include "caffe/util/nccl.hpp"
#endif

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline size_t mask_size() const {
    return mask_size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }
  inline Dtype* mask() const {
    return mask_;
  }

 protected:
  const size_t size_;           // Size of buffers
  const size_t mask_size_;      // Size of mask buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient
  Dtype* mask_;                 // Mask

DISABLE_COPY_AND_ASSIGN(Params);
};

// Params stored in GPU memory.
template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~GPUParams();

  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::mask_size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::mask_;

 private:
  int buffer_device_;
#ifndef CPU_ONLY
  cudaStream_t stream_;
#endif
};

class DevicePair {
 public:
  DevicePair(int parent, int device)
      : parent_(parent),
        device_(device) {
  }
  inline int parent() {
    return parent_;
  }
  inline int device() {
    return device_;
  }

  // Group GPUs in pairs, by proximity depending on machine's topology
  static void compute(const vector<int> devices, vector<DevicePair>* pairs);

 protected:
  int parent_;
  int device_;
};

// Scores Holder
template<typename Dtype>
struct SharedScores {
  explicit SharedScores(size_t nranks) : memory_(nranks) {
    for (size_t i = 0; i < nranks; ++i) {
      memory_[i].resize(MAX_SCORES);
    }
  }
  vector<Dtype>& rank_scores(size_t rank) {
    CHECK_LT(rank, memory_.size());
    return memory_[rank];
  }
  static constexpr size_t MAX_SCORES = 1000;

 private:
  vector<vector<Dtype>> memory_;
};

template<typename Dtype>
constexpr size_t SharedScores<Dtype>::MAX_SCORES;

// Synchronous data parallelism using map-reduce between local GPUs.
template<typename Dtype>
class P2PSync : public GPUParams<Dtype>, public Solver<Dtype>::Callback,
    public InternalThread {
 public:
  explicit P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                   int rank, int nranks, const SolverParameter& param);
  virtual ~P2PSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run(const vector<int>& gpus);
  // void Prepare(const vector<int>& gpus,
  //              vector<shared_ptr<P2PSync<Dtype> > >* syncs);
  inline int initial_iter() const { return initial_iter_; }

  // Divide the batch size by the number of solvers
  static void divide_batch_size(NetParameter* net);

#ifdef USE_NCCL
  // set the NCCL communicator
  void setNCCLComm(ncclComm_t comm);
#endif

 public:
  void allreduce(int param_id);
  void syncCommStream();
  void saveTestResults(float loss, const vector<float>& scores) override;
  void aggregateTestResults(float* loss, vector<float>* scores) override;

 protected:
  void SetupP2PAccess();
  void soft_barrier();
  void sync_data();
  void on_start();
  void allreduce();
  void syncAllStreams();
#ifndef CPU_ONLY
#ifdef USE_NCCL
  ncclComm_t getNCCLComm();
#endif
  cudaStream_t getCommStream();
#endif
  void InternalThreadEntry();

  const int rank_;
  const int nranks_;
  P2PSync<Dtype>* parent_;
  vector<P2PSync<Dtype>*> children_;
#ifndef CPU_ONLY
#ifdef USE_NCCL
  std::vector<ncclComm_t> nccl_comms_;
#endif
  vector<cudaStream_t> comm_streams_;
#endif
  BlockingQueue<P2PSync<Dtype>*> queue_;
  const int initial_iter_;

  shared_ptr<Solver<Dtype> > solver_;
  const SolverParameter& params_;

  // per-parameter reduction enabled
  bool per_parameter_reduce_;

  using Params<Dtype>::size_;
  using Params<Dtype>::mask_size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::mask_;

  // memory shared between threads
  shared_ptr<SharedScores<float> > shared_;
};

}  // namespace caffe

#endif

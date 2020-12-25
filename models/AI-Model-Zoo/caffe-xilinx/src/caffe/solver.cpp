#include <cstdio>

#include <string>
#include <vector>

#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/util/bbox_util.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false), iteration_timer_(), iterations_last_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false), iteration_timer_(), iterations_last_() {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
      << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if ((param_.eval_type() != "detection" && param_.eval_type() != "densebox") || Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  total_lapse_ = 0.f;
  current_step_ = 0;
  if (!Caffe::root_solver() && root_solver_) {
    iter_ = root_solver_->iter_;
    current_step_ = root_solver_->current_step_;
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  // CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

  net_->SetSolver(this);

  // Test on multi-gpu
#ifndef CPU_ONLY
  if (Caffe::solver_count() > 1) {
    for (int i = 0; i < callbacks_.size(); ++i) {
      // we need to sync all threads before starting, otherwise some cuda init,
      // malloc or other cuda stuff could interlock with in-loop cuda GPU sync
      // called in on_start.
      callbacks_[i]->soft_barrier();
      // Initial bcast of parameters
      callbacks_[i]->on_start();
    }
    LOG(INFO) << "Starting Optimizatoin on GPU " << Caffe::current_device();
  }
  const bool use_multi_gpu_testing = Caffe::solver_count() > 1;
  const string mgpu_str = use_multi_gpu_testing ? "[MultiGPU] " : "";
#else
  const bool use_multi_gpu_testing = false;
  const string mgpu_str;
#endif

  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      // if (Caffe::root_solver()) {
        TestAll(use_multi_gpu_testing);
      // }
      callback_soft_barrier();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();

    if (requested_early_exit_) {
      total_lapse_ += iteration_timer_.Seconds();
      break;
    }

    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      float lapse = iteration_timer_.Seconds();
      total_lapse_ += lapse;
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      unsigned int remaining_minutes = static_cast<int>(
                                   (stop_iter - iter_) / 60 / (per_s + 0.00001));
      unsigned int remaining_hours = remaining_minutes / 60;
      remaining_minutes %= 60;
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() <<" iter), loss = " << smoothed_loss_
          <<", remaining " << remaining_hours 
          << " hours and " << remaining_minutes << " minutes";
      iteration_timer_.Start();
      iterations_last_ = iter_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
#ifndef CPU_ONLY
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
#endif
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->allreduce();
    }
    // Make sure all gradient exchanges have finished in per-level scheme
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->syncCommStream();
    }

    //LOG(INFO)<< "root_sover:" << Caffe::root_solver();
    //if (Caffe::root_solver()) SnapshotToBinaryProto();
    ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      total_lapse_ += iteration_timer_.Seconds();
      // Break out of training loop.
      break;
    }
  }
}

float perf(int iter, float lapse) {
  return (iter > 0 ? iter : 0) / (lapse < 0.1 ? 0.1 : lapse);
}

template <typename Dtype>
int Solver<Dtype>::Solve(const char* resume_file) {
  // CHECK(Caffe::root_solver());
  LOG_IF(INFO, Caffe::root_solver()) << "Solving " << net_->name();
  LOG_IF(INFO, Caffe::root_solver()) << "Learning Rate Policy: "
      << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  callback_soft_barrier();

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (Caffe::root_solver() && param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG_IF(INFO, Caffe::root_solver()) << "Optimization stopped early (" <<
        perf(iter_, total_lapse_) << " iter/s).";
    return -1; // Early stop
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->set_phase(TEST);
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG_IF(INFO, Caffe::root_solver()) << "Iteration "
        << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    bool use_multi_gpu_testing = Caffe::solver_count() > 1;
    TestAll(use_multi_gpu_testing);
    callback_soft_barrier();
  }
  LOG_IF(INFO, Caffe::root_solver()) << "Optimization Done (" <<
      perf(iter_, total_lapse_) << " iter/s).";

  return 0; // Normal exit
}

template <typename Dtype>
void Solver<Dtype>::TestAll(bool use_multi_gpu) {
  // we need to sync all threads before testing, otherwise some cuda init,
  // malloc or other cuda stuff could interlock with in-loop cuda GPU sync
  callback_soft_barrier();
  // Sync bcast of parameters
  callback_sync_data();

  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    if (param_.eval_type() == "detection") {
      if (Caffe::root_solver()) TestSSD(test_net_id);
    } else if (param_.eval_type() == "segmentation") {
      TestSegmentation(test_net_id, use_multi_gpu);
    } else if (param_.eval_type() == "densebox") {
      if (Caffe::root_solver()) TestDenseBox(test_net_id);
    } else {
      Test(test_net_id, use_multi_gpu);
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id, bool use_multi_gpu) {
  // CHECK(Caffe::root_solver());
  LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  callback_soft_barrier();
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<float> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  float loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      LOG_IF(INFO, Caffe::root_solver()) << "Test interrupted.";
      return;
      // break out of test loop.
      // break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  // if (requested_early_exit_) {
  //   LOG(INFO) << "Test interrupted.";
  //   return;
  // }

  if (use_multi_gpu && test_score.size() <= SharedScores<Dtype>::MAX_SCORES) {
    callback_soft_barrier();
    // now we've done, transfer results
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->saveTestResults(loss, test_score);
    }
    callback_soft_barrier();
    float global_loss = 0.F;
    vector<float> global_scores(test_score.size());
    // aggregate test results from all solvers
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->aggregateTestResults(&global_loss, &global_scores);
    }
    callback_soft_barrier();
    loss = global_loss;
    for (int i = 0; i < test_score.size(); ++i) {
      test_score[i] = global_scores[i];
    }
  }

  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }

  if (use_multi_gpu && test_score.size() > SharedScores<Dtype>::MAX_SCORES) {
    LOG(WARNING) << "[MultiGPU] Test net output too much values: " <<
        test_score.size();
  }

  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id) /
        (test_score.size() > SharedScores<Dtype>::MAX_SCORES ?
            1.0 : Caffe::solver_count());
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG_IF(INFO, Caffe::root_solver()) << "    Test net output #" << i <<
        ": " << output_name << " = " << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::TestSegmentation(const int test_net_id, bool use_multi_gpu) {

  // CHECK(Caffe::root_solver());
  LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
                                     << ", Testing net (#" << test_net_id << ")";
  callback_soft_barrier();
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
          ShareTrainedLayersWith(net_.get());

  int class_num = param_.classiou_class_num();
  std::vector<u_int64_t> tp_;
  std::vector<u_int64_t> fp_;
  std::vector<u_int64_t> fn_;
  tp_.resize(class_num, 0);
  fp_.resize(class_num, 0);
  fn_.resize(class_num, 0);
  vector<float> iou(class_num);

  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  std::vector<float> output(class_num * 3, 0);
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
      if (SolverAction::SNAPSHOT == request) {
        Snapshot();
      } else if (SolverAction::STOP == request) {
        requested_early_exit_ = true;
      }
      request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      LOG_IF(INFO, Caffe::root_solver()) << "Test interrupted.";
      return;
      // break out of test loop.
      // break;
    }

    // Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(nullptr);
    // if (param_.test_compute_loss()) {
    //   loss += iter_loss;
    // }
    for (int j = 0; j < result.size(); j++) {
      if (result[j]->num() == 3 && result[j]->channels() == class_num) {
        const Dtype *result_vec = result[j]->cpu_data();
        for (int c = 0; c < class_num * 3; ++c) {
          output[c] += result_vec[c];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }

  if (use_multi_gpu) {
    callback_soft_barrier();
    // now we've done, transfer results
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->saveTestResults(loss, output);
    }
    callback_soft_barrier();

    float global_loss = 0.F;
    std::vector<float> global_output(output.size());
    // aggregate test results from all solvers
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->aggregateTestResults(&global_loss, &global_output);
    }
    callback_soft_barrier();
    for (int i = 0; i < output.size(); ++i) {
      output[i] = global_output[i];
    }
  }

  for (int c = 0; c < class_num; c++) {
    tp_[c] = static_cast<int>(output[c]);
    fp_[c] = static_cast<int>(output[class_num + c]);
    fn_[c] = static_cast<int>(output[class_num * 2 + c]);
  }
  // if (param_.test_compute_loss()) {
  //   loss /= param_.test_iter(test_net_id);
  //   LOG(INFO) << "Test loss: " << loss;
  // }
  for (int c = 0; c < class_num; ++c) {
    u_int64_t denom = tp_[c] + fp_[c] + fn_[c];
    if (denom == 0) {
      iou[c] = NAN;
    } else {
      iou[c] = tp_[c] * 1. / denom;
    }
  }

  // average IOU on C classes
  float mIOU = 0;
  int count = 0;
  for (int c = 0; c < class_num; ++c) {
    if (!std::isnan(iou[c])) {
      mIOU += iou[c];
      count++;
    }
  }
  if (count == 0) {
    mIOU = NAN;
  } else {
    mIOU = mIOU / count;
  }
  LOG_IF(INFO, Caffe::root_solver()) << "    Test net output #" << 0 << ": segmentation_eval_classIOU = " << mIOU;
}


template <typename Dtype>
void Solver<Dtype>::TestDenseBox(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  map<int, map<int, vector<pair<float, int> > > > all_true_pos;
  map<int, map<int, vector<pair<float, int> > > > all_false_pos;
  map<int, map<int, int> > all_num_pos;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  // Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    // Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(nullptr);
    // if (param_.test_compute_loss()) {
    //   loss += iter_loss;
    // }
    for (int j = 0; j < result.size(); ++j) {
      // CHECK_EQ(result[j]->width(), 5);
      const Dtype* result_vec = result[j]->cpu_data();
      int num_det = result[j]->height();
      if (result[j]->width() != 5) {
        continue;  
      }
      CHECK_EQ(result[j]->width(), 5);
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);
        int label = static_cast<int>(result_vec[k * 5 + 1]);
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];
          int tp = static_cast<int>(result_vec[k * 5 + 3]);
          int fp = static_cast<int>(result_vec[k * 5 + 4]);
          if (tp == 0 && fp == 0) {
            // Ignore such case. It happens when a detection bbox is matched to
            // a difficult gt bbox and we don't evaluate on difficult gt bbox.
            continue;
          }
          all_true_pos[j][label].push_back(std::make_pair(score, tp));
          all_false_pos[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  // if (param_.test_compute_loss()) {
  //   loss /= param_.test_iter(test_net_id);
  //   LOG(INFO) << "Test loss: " << loss;
  // }
  for (int i = 0; i < all_true_pos.size(); ++i) {
    if (all_true_pos.find(i) == all_true_pos.end()) {
      LOG(FATAL) << "Missing output_blob true_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& true_pos =
        all_true_pos.find(i)->second;
    if (all_false_pos.find(i) == all_false_pos.end()) {
      LOG(FATAL) << "Missing output_blob false_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& false_pos =
        all_false_pos.find(i)->second;
    if (all_num_pos.find(i) == all_num_pos.end()) {
      LOG(FATAL) << "Missing output_blob num_pos: " << i;
    }
    const map<int, int>& num_pos = all_num_pos.find(i)->second;
    map<int, float> APs;
    float mAP = 0.;
    // Sort true_pos and false_pos with descend scores.
    for (map<int, int>::const_iterator it = num_pos.begin();
         it != num_pos.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (true_pos.find(label) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_true_pos =
          true_pos.find(label)->second;
      if (false_pos.find(label) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_false_pos =
          false_pos.find(label)->second;
      vector<float> prec, rec;
      ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                param_.ap_version(), &prec, &rec, &(APs[label]));
      mAP += APs[label];
      if (param_.show_per_class_result()) {
        LOG(INFO) << "class" << label << ": " << APs[label];
      }
    }
    mAP /= num_pos.size();
    const int output_blob_index = test_net->output_blob_indices()[i];
    const string& output_name = test_net->blob_names()[output_blob_index];
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mAP;
  }
}


template <typename Dtype>
void Solver<Dtype>::TestSSD(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  map<int, map<int, vector<pair<float, int> > > > all_true_pos;
  map<int, map<int, vector<pair<float, int> > > > all_false_pos;
  map<int, map<int, int> > all_num_pos;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  // Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    // Dtype iter_loss;
    const vector<Blob<Dtype>*>& result = test_net->Forward(nullptr);
    // if (param_.test_compute_loss()) {
    //   loss += iter_loss;
    // }
    for (int j = 0; j < result.size(); ++j) {
      CHECK_EQ(result[j]->width(), 5);
      const Dtype* result_vec = result[j]->cpu_data();
      int num_det = result[j]->height();
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);
        int label = static_cast<int>(result_vec[k * 5 + 1]);
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];
          int tp = static_cast<int>(result_vec[k * 5 + 3]);
          int fp = static_cast<int>(result_vec[k * 5 + 4]);
          if (tp == 0 && fp == 0) {
            // Ignore such case. It happens when a detection bbox is matched to
            // a difficult gt bbox and we don't evaluate on difficult gt bbox.
            continue;
          }
          all_true_pos[j][label].push_back(std::make_pair(score, tp));
          all_false_pos[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  // if (param_.test_compute_loss()) {
  //   loss /= param_.test_iter(test_net_id);
  //   LOG(INFO) << "Test loss: " << loss;
  // }
  for (int i = 0; i < all_true_pos.size(); ++i) {
    if (all_true_pos.find(i) == all_true_pos.end()) {
      LOG(FATAL) << "Missing output_blob true_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& true_pos =
        all_true_pos.find(i)->second;
    if (all_false_pos.find(i) == all_false_pos.end()) {
      LOG(FATAL) << "Missing output_blob false_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& false_pos =
        all_false_pos.find(i)->second;
    if (all_num_pos.find(i) == all_num_pos.end()) {
      LOG(FATAL) << "Missing output_blob num_pos: " << i;
    }
    const map<int, int>& num_pos = all_num_pos.find(i)->second;
    map<int, float> APs;
    float mAP = 0.;
    // Sort true_pos and false_pos with descend scores.
    for (map<int, int>::const_iterator it = num_pos.begin();
         it != num_pos.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (true_pos.find(label) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_true_pos =
          true_pos.find(label)->second;
      if (false_pos.find(label) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_false_pos =
          false_pos.find(label)->second;
      vector<float> prec, rec;
      ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                param_.ap_version(), &prec, &rec, &(APs[label]));
      mAP += APs[label];
      if (param_.show_per_class_result()) {
        LOG(INFO) << "class" << label << ": " << APs[label];
      }
    }
    mAP /= num_pos.size();
    const int output_blob_index = test_net->output_blob_indices()[i];
    const string& output_name = test_net->blob_names()[output_blob_index];
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mAP;
  }
}
   
template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff(), true);
  // net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

template <typename Dtype>
void Solver<Dtype>::callback_soft_barrier() {
  for (int i = 0; i < callbacks_.size(); ++i) {
    callbacks_[i]->soft_barrier();
  }
}

template <typename Dtype>
void Solver<Dtype>::callback_sync_data() {
  for (int i = 0; i < callbacks_.size(); ++i) {
    callbacks_[i]->sync_data();
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe

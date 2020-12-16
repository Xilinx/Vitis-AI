#include <boost/thread.hpp>
#ifndef USE_BOOST
#include <mutex>
#endif
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <>
map<const string, weak_ptr<DataReader<string>::Body> >
    DataReader<string>::bodies_
    = map<const string, weak_ptr<DataReader<string>::Body> >();
template <>
map<const string, weak_ptr<DataReader<Datum>::Body> >
    DataReader<Datum>::bodies_
    = map<const string, weak_ptr<DataReader<Datum>::Body> >();
template <>
map<const string, weak_ptr<DataReader<AnnotatedDatum>::Body> >
    DataReader<AnnotatedDatum>::bodies_
    = map<const string, weak_ptr<DataReader<AnnotatedDatum>::Body> >();
template <>
map<const string, weak_ptr<DataReader<DrivingData>::Body> >
    DataReader<DrivingData>::bodies_
    = map<const string, weak_ptr<DataReader<DrivingData>::Body> >();
template <>
map<const string, weak_ptr<DataReader<DriveData>::Body> >
    DataReader<DriveData>::bodies_
    = map<const string, weak_ptr<DataReader<DriveData>::Body> >();
template <>
map<const string, weak_ptr<DataReader<DirectRegressionData>::Body> >
    DataReader<DirectRegressionData>::bodies_
    = map<const string, weak_ptr<DataReader<DirectRegressionData>::Body> >();

#ifdef USE_BOOST
static boost::mutex bodies_mutex_;
#else
static std::mutex bodies_mutex_;
#endif

template <typename T>
DataReader<T>::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
#ifdef USE_BOOST
  boost::mutex::scoped_lock lock(bodies_mutex_);
#else
  std::lock_guard<std::mutex> lock(bodies_mutex_);
#endif
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

template <typename T>
DataReader<T>::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
#ifdef USE_BOOST
  boost::mutex::scoped_lock lock(bodies_mutex_);
#else
  std::lock_guard<std::mutex> lock(bodies_mutex_);
#endif
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

template <typename T>
DataReader<T>::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new T());
  }
}

template <typename T>
DataReader<T>::QueuePair::~QueuePair() {
  T* t;
  while (free_.try_pop(&t)) {
    delete t;
  }
  while (full_.try_pop(&t)) {
    delete t;
  }
}

template <typename T>
DataReader<T>::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

template <typename T>
DataReader<T>::Body::~Body() {
  StopInternalThread();
}

template <typename T>
void DataReader<T>::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  try {
    // int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;
    bool mgpu_testing = param_.data_param().multi_gpu_testing();
    bool single_solver = (param_.phase() == TEST) && (!mgpu_testing);
    int solver_count = single_solver ? 1 : Caffe::solver_count();
    LOG(INFO) << (param_.phase() == TEST ? "TEST: " : "TRAIN: ")
              << "reading data using " << solver_count << " channel(s)";

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(cursor.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

template <typename T>
void DataReader<T>::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  T* t = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  t->ParseFromString(cursor->value());
  qp->full_.push(t);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

template <>
void DataReader<string>::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  string* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  *datum = cursor->value();
  qp->full_.push(datum);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

// Instance class
template class DataReader<string>;
template class DataReader<Datum>;
template class DataReader<AnnotatedDatum>;
template class DataReader<DrivingData>;
template class DataReader<DriveData>;
template class DataReader<DirectRegressionData>;
}  // namespace caffe

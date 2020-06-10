/*
* Copyright 2019 Xilinx Inc.
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

// vai_q_caffe.cpp: the complete release version of VAI_Q_CAFFE

#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <map>
#include <string>
#include <unordered_set>
#include <vector>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/convert_proto.hpp"
#include "caffe/util/fix_pos_check.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/net_test.hpp"
#include "caffe/util/signal_handler.h"

using namespace caffe; // NOLINT(build/namespaces)
using namespace std;
using namespace boost::filesystem;

DEFINE_string(model, "", "The model definition protocol buffer text file");
DEFINE_string(weights, "", "The pretrained float weights for quantization");
DEFINE_string(solver, "", "The solver for finetuning or training");
DEFINE_string(output_dir, "./quantize_results",
              "Optional: Output directory for the quantization results");
#ifdef CPU_ONLY
DEFINE_string(gpu, "", "Optional: The GPU device id for calibration and test");
#else
DEFINE_string(gpu, "0", "Optional: The GPU device id for calibration and test");
#endif
DEFINE_int32(method, 1,
             "Optional: Method for quantization, 0: Non-Overflow, 1: Min-Diff");
DEFINE_int32(data_bit, 8,
             "Optional: Bit width for activation data during quantization");
DEFINE_int32(weights_bit, 8,
             "Optional: Bit width for weights and bias during quantization");
DEFINE_int32(calib_iter, 100,
             "Optional: Total iterations for quantize calibration");
DEFINE_bool(auto_test, false,
            "Optional: Whether to test after calibration, need test dataset");
DEFINE_int32(test_iter, 50, "Optional: Total iterations for test");
// DEFINE_bool(keep_bn, false,
// "Optional: Remain BatchNorm layers during quantization");
// DEFINE_bool(keep_convbn, false,
// "Optional: remain ConvolutinBNFixed layers instead of ConvolutionFixed");
DEFINE_string(ignore_layers, "", "Optinal: List of layers to be ignore during "
                                 "quantization, comma-delimited");
DEFINE_string(
    special_layers, "",
    "Optinal: List of layers which has special data_bit or weight_bit for "
    "quantization, comma-delimited, e.g. 'data_in,16,conv2,8,16', "
    "note:(1) the first number after layer_name represent data_bit, and the "
    "second "
    "number represent weight_bit(optional, can be omitted). (2) We "
    "have take some necessary measures to check the input parameters");
DEFINE_string(
    ignore_layers_file, "",
    "Optional: Prototxt file which defines the layers to be ignore during "
    "quantization, each line formated as 'ignore_layer: \"layer_name\"'");
DEFINE_string(input_blob, "data", "Optional: Name of input data blob");
DEFINE_string(sigmoided_layers, "", "Optinal: List of layers before sigmoid "
                                    "operation, to be quantized with "
                                    "optimization "
                                    "for sigmoid accuracy");
DEFINE_string(net_type, "", "Optional: Net types for net testing, including: "
                            "classification, detection, segmentation, other");

// Detection Net Params
DEFINE_string(ap_style, "11point",
              "Optional: AP computing styles for detection net, including: "
              "11point, MaxIntegral, Integral");

// Segmentation Net Params
DEFINE_string(seg_metric, "classIOU", "Optional; Test metric for segmentation "
                                      "net, now only classIOU is supported");
DEFINE_int32(seg_class_num, 19, "Optional; The number of segmentation classes");
DEFINE_bool(keep_fixed_neuron, false,
                        "Remain FixedNeuron layers in the output float caffemodel ");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func)                                             \
  namespace {                                                                  \
  class __Registerer_##func {                                                  \
  public: /* NOLINT */                                                         \
    __Registerer_##func() { g_brew_map[#func] = &func; }                       \
  };                                                                           \
  __Registerer_##func g_registerer_##func;                                     \
  }

static BrewFunction GetBrewFunction(const caffe::string &name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin(); it != g_brew_map.end();
         ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL; // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int> *gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Calibration activation for FixedNeuronLayer
NetParameter calib(NetParameter net_param, NetParameter &weights,
                   int max_iter) {
  // MultiBoxLoss layer has bugs during forward, remove it before calibration.
  net_param = RemoveLayerByType(net_param, "MultiBoxLoss");
  net_param = RemoveLayerByType(net_param, "MultiBoxLossRefine");

  // CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  net_param.mutable_state()->set_phase(TRAIN);

  Net<float> net(net_param);
  net.CopyTrainedLayersFrom(weights);

  int iter = 0;
  float iter_loss;
  // net.set_piter(&iter);
  for (auto &layer : net.layers()) {
    layer->set_piter(&iter);
  }

  LOG(INFO) << "Start Calibration";
  /*
  for (int i = 0; i < max_iter; i++) {
    net.Forward(&iter_loss);
    LOG(INFO) << "Calibration iter: " << iter + 1 << "/" << max_iter
              << " ,loss: " << iter_loss;
    iter++;
  }
  */
  int i = 0;
  // In DPU, concat/DeephiResize layer needs all its bottom data
  // fix position allign to top data
  bool fixPosConverge = true;
  for (auto &layer : net.layers()) {
    layer->set_pos_converge(&fixPosConverge);
  }
  while (i < max_iter || !fixPosConverge) {
    net.Forward(&iter_loss);
    // if ( i >= max_iter - 1 )
    //  fixPosConverge = checkPosConverge( net );
    fixPosConverge = true;
    if (!fixPosConverge)
      LOG(INFO) << "Concat layers quantization position does not converge, do "
                   "quantization one more time!";
    LOG(INFO) << "Calibration iter: " << iter + 1 << "/" << max_iter
              << " ,loss: " << iter_loss;
    ++iter;
    ++i;
  }
  LOG(INFO) << "Calibration Done!";

  NetParameter calib_weights;
  net.ToProto(&calib_weights);
  return calib_weights;
}

// check dpu support for model
int check() {

  CHECK_GT(FLAGS_model.size(), 0) << "Need a float model to quantize.";

  // 0.Input
  NetParameter model_in;
  ReadNetParamsFromTextFileOrDie(string(FLAGS_model), &model_in);

  string check_results_file = "check_results.txt";
  bool pass = CheckModelAndSave(model_in, check_results_file);
  if (pass) {
    std::cout << "Check Passed!" << std::endl;
  } else {
    std::cout << "Check Failed, Please check the results file: "
              << check_results_file << "." << std::endl;
  }
  return 0;
}
RegisterBrewFunction(check);

// Quantize a float model and deploy
int quantize() {

  CHECK_GT(FLAGS_model.size(), 0) << "Need a float model to quantize.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need float weights to quantize.";

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif
  // Set mode and device id[s]
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  }

  path dir(FLAGS_output_dir);
  if (!exists(dir)) {
    create_directory(dir);
  }
  // 1.Input
  NetParameter model_in, weights_in;
  ReadNetParamsFromTextFileOrDie(string(FLAGS_model), &model_in);
  ReadNetParamsFromBinaryFileOrDie(string(FLAGS_weights), &weights_in);
  model_in = RemoveLayerByType(model_in, "Dropout");
  model_in = RemoveDuplicatedLayerByType(model_in, "BatchNorm");
  weights_in = RemoveLayerByType(weights_in, "Dropout");
  weights_in = RemoveDuplicatedLayerByType(weights_in, "BatchNorm");

  // Update bvlc batchnorm in model and weights
  NetParameter updated_model, updated_weights;
  bool keep_bn = true;
  MergeLayers(model_in, updated_model, false, keep_bn, false);
  MergeLayers(weights_in, updated_weights, true, keep_bn, false);

#if 0
  // Create a Net<float> and copy trained layers from weights to deal with
  // mismatch of model and weights.
  updated_model.mutable_state()->set_phase(TRAIN);
  Net<float> caffe_net(updated_model);
  caffe_net.CopyTrainedLayersFrom(updated_weights);
  caffe_net.ToProto(&updated_weights);
  CheckModel(updated_weights);
  DLOG(INFO) << "net loaded. ";
#else
  // multiple caffe::Net objects can cause LMDB open problem
  MergeModelAndWeights(updated_model, updated_weights);
#endif

  // 1.merge_model and prototxt
  NetParameter merged_model, merged_weights;
  keep_bn = false;
  MergeLayers(updated_model, merged_model, false, keep_bn, false);
  MergeLayers(updated_weights, merged_weights, true, keep_bn, false);

  // 2.convert_float prototxt to quantized prototxt
  NetParameter quantized_model;
  bool add_bn = false;
  ConvertFloat2Fix(merged_model, quantized_model, FLAGS_ignore_layers,
                   FLAGS_ignore_layers_file, FLAGS_method, FLAGS_weights_bit,
                   FLAGS_data_bit, add_bn, FLAGS_sigmoided_layers,
                   FLAGS_special_layers);
  path quantized_model_file = dir / path("quantize_train_test.prototxt");
  WriteProtoToTextFile(quantized_model, quantized_model_file.string());

  // 3.quantize weights & calibration for activation quantize
  int calib_iter = FLAGS_calib_iter;
  NetParameter quantized_weights =
      calib(quantized_model, merged_weights, calib_iter);
  quantized_weights = RemoveLayerByType(quantized_weights, "Split");
  path quantized_weights_file = dir / path("quantize_train_test.caffemodel");
  WriteProtoToBinaryFile(quantized_weights, quantized_weights_file.string());

  // extract input information
  NetParameter deploy_model, deploy_weights;
  vector<int> input_shape;
  input_shape = GetInputShape(quantized_model, FLAGS_input_blob);
  int input_layer_id = GetLastInputLayer(merged_model);
  string input_layer_name = merged_model.layer(input_layer_id).name();

  LayerParameter *input_layer = deploy_model.add_layer();
  input_layer->set_name(input_layer_name);
  input_layer->set_type("Input");
  input_layer->add_top(FLAGS_input_blob);
  InputParameter *input_param = input_layer->mutable_input_param();
  BlobShape *blob_shape = input_param->add_shape();
  for (int i = 0; i < input_shape.size(); i++) {
    blob_shape->add_dim(input_shape[i]);
  }
  blob_shape->set_dim(0, 1);
  *(input_layer->mutable_transform_param()) =
      GetTransformParam(quantized_model);

  // 4.test
  if (FLAGS_auto_test) {
    const string net_type =
        FLAGS_net_type.size() ? FLAGS_net_type : InferNetType(quantized_model);
    Net<float> quantized_net(quantized_model);
    quantized_net.CopyTrainedLayersFrom(quantized_weights);
    test_net_by_type(quantized_net, net_type, FLAGS_test_iter, FLAGS_ap_style,
                     FLAGS_seg_metric, FLAGS_seg_class_num);
  }

  // 5.deploy
  LOG(INFO) << "Start Deploy";
  FIX_INFO weight_infos, bias_infos, data_infos;
  TransformFix2FloatWithFixInfo(quantized_weights, deploy_weights, weight_infos,
                                bias_infos, data_infos, FLAGS_keep_fixed_neuron);
  vector<vector<int>> quantize_infos =
      GetFixInfo(deploy_weights, weight_infos, bias_infos, data_infos);
  deploy_model.MergeFrom(RemoveBlobs(deploy_weights));
  deploy_model = ConvertTrain2Deploy(deploy_model);
  LOG(INFO) << "Deploy Done!";

  // 6.output
  path quantize_info_file = dir / path("quantize_info.txt");
  path deploy_weights_file = dir / path("deploy.caffemodel");
  path deploy_model_file = dir / path("deploy.prototxt");

  std::cout << string(50, '-') << std::endl;
  std::cout << "Output Quantized Train&Test Model:   " << quantized_model_file
            << std::endl;
  std::cout << "Output Quantized Train&Test Weights: " << quantized_weights_file
            << std::endl;
  WriteFixInfoToModel(deploy_weights, quantize_infos);
  WriteProtoToBinaryFile(deploy_weights, deploy_weights_file.string());
  std::cout << "Output Deploy Weights: " << deploy_weights_file << std::endl;
  WriteProtoToTextFile(deploy_model, deploy_model_file.string());
  std::cout << "Output Deploy Model:   " << deploy_model_file << std::endl;
  if (getenv("DECENT_DEBUG") && stoi(getenv("DECENT_DEBUG")) > 0) {
    WriteFixInfoToFile(deploy_weights, quantize_infos,
                       quantize_info_file.string());
    std::cout << "Output Quantize Info for debugging: " << quantize_info_file
              << std::endl;
  }
  return 0;
}
RegisterBrewFunction(quantize);

// Deploy to DPU:1.quantize_info.txt 2.deploy.prototxt 3.deploy.caffemodel
int deploy() {

  CHECK_GT(FLAGS_model.size(), 0) << "Need a quantized model for deploy.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need quantized weights for deploy.";

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif
  // Set mode and device id[s]
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  }

  path dir(FLAGS_output_dir);
  if (!exists(dir)) {
    create_directory(dir);
  }

  NetParameter quantized_model, quantized_weights;
  ReadNetParamsFromTextFileOrDie(string(FLAGS_model), &quantized_model);
  ReadNetParamsFromBinaryFileOrDie(string(FLAGS_weights), &quantized_weights);
  quantized_weights = RemoveLayerByType(quantized_weights, "Split");
  DLOG(INFO) << "net loaded: ";

  // extract input information
  NetParameter deploy_model, deploy_weights;
  vector<int> input_shape;
  input_shape = GetInputShape(quantized_model, FLAGS_input_blob);
  int input_layer_id = GetLastInputLayer(quantized_model);
  string input_layer_name = quantized_model.layer(input_layer_id).name();

  LayerParameter *input_layer = deploy_model.add_layer();
  input_layer->set_name(input_layer_name);
  input_layer->set_type("Input");
  input_layer->add_top(FLAGS_input_blob);
  InputParameter *input_param = input_layer->mutable_input_param();
  BlobShape *blob_shape = input_param->add_shape();
  for (int i = 0; i < input_shape.size(); i++) {
    blob_shape->add_dim(input_shape[i]);
  }
  blob_shape->set_dim(0, 1);
  *(input_layer->mutable_transform_param()) =
      GetTransformParam(quantized_model);

  LOG(INFO) << "Start Deploy";
  FIX_INFO weight_infos, bias_infos, data_infos;
  TransformFix2FloatWithFixInfo(quantized_weights, deploy_weights, weight_infos,
                                bias_infos, data_infos, FLAGS_keep_fixed_neuron);
  vector<vector<int>> quantize_infos =
      GetFixInfo(deploy_weights, weight_infos, bias_infos, data_infos);
  deploy_model.MergeFrom(RemoveBlobs(deploy_weights));
  deploy_model = ConvertTrain2Deploy(deploy_model);
  LOG(INFO) << "Deploy Done!";

  // 5.output
  path quantize_info_file = dir / path("quantize_info.txt");
  path deploy_weights_file = dir / path("deploy.caffemodel");
  path deploy_model_file = dir / path("deploy.prototxt");

  WriteFixInfoToModel(deploy_weights, quantize_infos);
  WriteProtoToBinaryFile(deploy_weights, deploy_weights_file.string());
  std::cout << "Output Deploy Weights: " << deploy_weights_file << std::endl;
  WriteProtoToTextFile(deploy_model, deploy_model_file.string());
  std::cout << "Output Deploy Model:   " << deploy_model_file << std::endl;
  if (getenv("DECENT_DEBUG") && stoi(getenv("DECENT_DEBUG")) > 0) {
    WriteFixInfoToFile(deploy_weights, quantize_infos,
                       quantize_info_file.string());
    std::cout << "Output Quantize Info For Debugging:       "
              << quantize_info_file << std::endl;
  }
  return 0;
}
RegisterBrewFunction(deploy);

// Train / Finetune a model.
int finetune() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0 &&
      solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    if (solver_param.has_device_id()) {
      FLAGS_gpu = "" + boost::lexical_cast<string>(solver_param.device_id());
    } else { // Set default GPU if unspecified
      FLAGS_gpu = "" + boost::lexical_cast<string>(0);
    }
  }

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif
  // Set mode and device id[s]
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(caffe::SolverAction::STOP,
                                      caffe::SolverAction::STOP);

  caffe::shared_ptr<caffe::Solver<float>> solver(
      caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_weights.size()) {
    solver->net()->CopyTrainedLayersFrom(FLAGS_weights);
  }

  if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, 0, gpus.size(), solver->param());
    sync.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";

  // solver.reset();
  return 0;
}
RegisterBrewFunction(finetune);

// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
  while (gpus.size() > 1) {
    // Only use one GPU
    LOG(INFO) << "Not using GPU #" << gpus.back() << " for single-GPU function";
    gpus.pop_back();
  }
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif

  // Set mode and device id
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // 0.Input
  NetParameter model_in, weights_in;
  ReadNetParamsFromTextFileOrDie(string(FLAGS_model), &model_in);
  ReadNetParamsFromBinaryFileOrDie(string(FLAGS_weights), &weights_in);

  // Update bvlc batchnorm in model and weights
  NetParameter updated_model, updated_weights;
  bool keep_bn = true;
  MergeLayers(model_in, updated_model, false, keep_bn, false);
  MergeLayers(weights_in, updated_weights, true, keep_bn, false);
  WriteProtoToTextFile(updated_model, "updated.prototxt");
  WriteProtoToBinaryFile(updated_weights, "updated.caffemodel");

  Net<float> test_net(updated_model);
  test_net.CopyTrainedLayersFrom(updated_weights);

  // 2.test the caffe net.
  const string net_type =
      FLAGS_net_type.size() ? FLAGS_net_type : InferNetType(updated_model);
  test_net_by_type(test_net, net_type, FLAGS_test_iter, FLAGS_ap_style,
                   FLAGS_seg_metric, FLAGS_seg_class_num);

  return 0;
}
RegisterBrewFunction(test);

string version_str() {
  std::stringstream ss;
  ss << "1.2.5\n"
     << "Build Label " << __DATE__ << " " << __TIME__ << "\n";
  return ss.str();
}

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  gflags::SetVersionString(version_str());
  gflags::SetUsageMessage(
      "command line brew\n"
      "usage: vai_q_caffe <command> <args>\n\n"
      "commands:\n"
      "  quantize        quantize a float model "
      "(need calibration with dataset) and deploy to DPU\n"
      "  deploy          deploy quantized model to DPU\n"
      "  finetune        train or finetune a model\n"
      "  test            score a model\n\n"
      "example:\n"
      "  1. quantize:                           ./vai_q_caffe quantize -model "
      "float.prototxt -weights float.caffemodel -gpu 0\n"
      "  2. quantize with auto test:            ./vai_q_caffe quantize -model "
      "float.prototxt -weights float.caffemodel -gpu 0 -auto_test -test_iter "
      "50\n"
      "  3. quantize with Non-Overflow method:  ./vai_q_caffe quantize -model "
      "float.prototxt -weights float.caffemodel -gpu 0 -method 0\n"
      "  4. finetune quantized model:           ./vai_q_caffe finetune -solver "
      "solver.prototxt -weights quantize_results/float_train_test.caffemodel "
      "-gpu 0\n"
      "  5. deploy quantized model:             ./vai_q_caffe deploy "
      "-model quantize_results/quantize_train_test.prototxt -weights "
      "quantize_results/float_train_test.caffemodel -gpu 0\n");

  // Run tool or show usage.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // google::InitGoogleLogging(argv[0]);

  if (argc == 2) {
    GetBrewFunction(caffe::string(argv[1]))();
    return 0;
  } else {
    ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/decent_q.cpp");
    return -1;
  }
}

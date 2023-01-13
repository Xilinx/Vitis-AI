/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/
%module vai_wrapper

%include <std_string.i>
/*
%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"
*/
%include "strings.i"
%include "base.i"


%{
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/stat_summarizer.h"
/*#include "tensorflow/python/lib/core/py_func.h"*/

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "graph_quantizer.h"
#include "fold_batch_norms.h"
#include "tf_status_helper.h"
%}

%ignoreall

%unignore tensorflow;
%unignore DecentQCheckGraphWithStringInputs;
%unignore DecentQConvertConstantsToVariablesWithStringInputs;
%unignore DecentQCreateOptimizedGraphWithStringInputs;
%unignore DecentQCreateQuantizeCalibrationGraphWithStringInputs;
%unignore DecentQCreateQuantizeTrainingGraphWithStringInputs;
%unignore DecentQCreateQuantizeEvaluationGraphWithStringInputs;
%unignore DecentQConvertFoldedBatchnormsWithStringInputs;
%unignore DecentQCreateQuantizeDeployGraphWithStringInputs;


%{

tensorflow::GraphDef StringToGraph(string graph_def_string, 
                                   TF_Status* out_status) {
  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_string)) {
    Set_TF_Status_from_Status(out_status, tensorflow::errors::InvalidArgument(
        "Couldn't interpret input as a GraphDef"));
    return graph_def;
  }
  return graph_def;
}

string GraphToString(tensorflow::GraphDef graph_def, TF_Status* out_status) {
  string result;
  if (!graph_def.SerializeToString(&result)) {
    Set_TF_Status_from_Status(out_status, tensorflow::errors::InvalidArgument(
        "Couldn't serialize output as a GraphDef"));
    return "";
  }
  return result;
}

void DecentQCheckGraphWithStringInputs(string graph_def_string, string graph_path,
                                         TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));

  tensorflow::decent_q::GraphQuantizer graph_quantizer;
  tensorflow::Status check_status =
      graph_quantizer.CheckGraph(input_graph_def, graph_path);
  if (!check_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, check_status);
    return;
  }

  return;
}


string DecentQConvertConstantsToVariablesWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));
  tensorflow::GraphDef output_graph_def;

  tensorflow::decent_q::QuantizeConfig config;
  tensorflow::Status config_status = config.FromString(config_string);
  if (!config_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, config_status);
    return "";
  }
  tensorflow::decent_q::GraphQuantizer graph_quantizer(config);
  tensorflow::Status create_status =
      graph_quantizer.ConvertConstantsToVariables(input_graph_def, 
          output_graph_def);
  if (!create_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, create_status);
    return "";
  }

  return GraphToString(output_graph_def, out_status);
}

string DecentQCreateOptimizedGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));
  tensorflow::GraphDef output_graph_def;

  tensorflow::decent_q::QuantizeConfig config;
  tensorflow::Status config_status = config.FromString(config_string);
  if (!config_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, config_status);
    return "";
  }
  tensorflow::decent_q::GraphQuantizer graph_quantizer(config);
  tensorflow::Status create_status =
      graph_quantizer.CreateOptimizedGraph(input_graph_def, 
          output_graph_def);
  if (!create_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, create_status);
    return "";
  }

  return GraphToString(output_graph_def, out_status);
}

string DecentQCreateQuantizeCalibrationGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));
  tensorflow::GraphDef output_graph_def;

  tensorflow::decent_q::QuantizeConfig config;
  tensorflow::Status config_status = config.FromString(config_string);
  if (!config_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, config_status);
    return "";
  }
  tensorflow::decent_q::GraphQuantizer graph_quantizer(config);
  tensorflow::Status create_status =
      graph_quantizer.CreateQuantizeCalibrationGraph(input_graph_def, 
          output_graph_def);
  if (!create_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, create_status);
    return "";
  }

  return GraphToString(output_graph_def, out_status);
}

string DecentQCreateQuantizeTrainingGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));
  tensorflow::GraphDef output_graph_def;

  tensorflow::decent_q::QuantizeConfig config;
  tensorflow::Status config_status = config.FromString(config_string);
  if (!config_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, config_status);
    return "";
  }
  tensorflow::decent_q::GraphQuantizer graph_quantizer(config);
  tensorflow::Status create_status =
      graph_quantizer.CreateQuantizeTrainingGraph(input_graph_def, 
          output_graph_def);
  if (!create_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, create_status);
    return "";
  }

  return GraphToString(output_graph_def, out_status);
}

string DecentQCreateQuantizeEvaluationGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));
  tensorflow::GraphDef output_graph_def;

  tensorflow::decent_q::QuantizeConfig config;
  tensorflow::Status config_status = config.FromString(config_string);
  if (!config_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, config_status);
    return "";
  }
  tensorflow::decent_q::GraphQuantizer graph_quantizer(config);
  tensorflow::Status create_status =
      graph_quantizer.CreateQuantizeEvaluationGraph(input_graph_def, 
          output_graph_def);
  if (!create_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, create_status);
    return "";
  }

  return GraphToString(output_graph_def, out_status);
}

string DecentQConvertFoldedBatchnormsWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));
  tensorflow::GraphDef output_graph_def;

  tensorflow::decent_q::QuantizeConfig config;
  tensorflow::Status config_status = config.FromString(config_string);
  if (!config_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, config_status);
    return "";
  }
  tensorflow::decent_q::GraphQuantizer graph_quantizer(config);
  tensorflow::Status create_status =
      graph_quantizer.ConvertFoldedBatchnorms(input_graph_def, 
          output_graph_def);
  if (!create_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, create_status);
    return "";
  }

  return GraphToString(output_graph_def, out_status);
}

string DecentQCreateQuantizeDeployGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status) {
  tensorflow::GraphDef input_graph_def(StringToGraph(graph_def_string, out_status));
  tensorflow::GraphDef output_graph_def;

  tensorflow::decent_q::QuantizeConfig config;
  tensorflow::Status config_status = config.FromString(config_string);
  if (!config_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, config_status);
    return "";
  }
  tensorflow::decent_q::GraphQuantizer graph_quantizer(config);
  tensorflow::Status create_status =
      graph_quantizer.CreateQuantizeDeployGraph(input_graph_def, 
          output_graph_def);
  if (!create_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, create_status);
    return "";
  }

  return GraphToString(output_graph_def, out_status);
}
%}


void DecentQCheckGraphWithStringInputs(string graph_def_string, string graph_path,
                                         TF_Status* out_status);
string DecentQConvertConstantsToVariablesWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status);
string DecentQCreateOptimizedGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status);
string DecentQCreateQuantizeCalibrationGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status);
string DecentQCreateQuantizeTrainingGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status);
string DecentQCreateQuantizeEvaluationGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status);
string DecentQConvertFoldedBatchnormsWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status);
string DecentQCreateQuantizeDeployGraphWithStringInputs(string graph_def_string,
                                      string config_string, TF_Status* out_status);

%unignoreall

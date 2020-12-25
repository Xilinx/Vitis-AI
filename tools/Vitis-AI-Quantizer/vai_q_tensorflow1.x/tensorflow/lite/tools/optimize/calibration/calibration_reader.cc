/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"

#include "absl/memory/memory.h"

namespace tflite {
namespace optimize {
namespace calibration {
TfLiteStatus CalibrationReader::GetTensorStatsAsMap(
    std::unordered_map<int, CalibrationStats>* tensor_id_to_stats_map) const {
  tensor_id_to_stats_map->clear();
  for (const auto& tensorid_stat : logger_->GetCalibrationValues()) {
    auto minmax = tensorid_stat.second;
    CalibrationReader::CalibrationStats stats;
    TF_LITE_ENSURE_STATUS(minmax.Get(&stats.min, &stats.max));
    tensor_id_to_stats_map->insert({tensorid_stat.first, stats});
  }

  return kTfLiteOk;
}

TfLiteStatus CalibrationReader::AddCalibrationToModel(ModelT* model,
                                                      bool update) const {
  if (!model || model->subgraphs.empty()) {
    return kTfLiteError;
  }
  const auto& subgraph = model->subgraphs[0];
  for (const auto& tensorid_stat : logger_->GetCalibrationValues()) {
    auto minmax = tensorid_stat.second;
    float min, max;
    TF_LITE_ENSURE_STATUS(minmax.Get(&min, &max));
    if (update) {
      auto tensor = subgraph->tensors[tensorid_stat.first].get();
      if (tensor->quantization) {
        const float existing_min = tensor->quantization->min[0];
        const float existing_max = tensor->quantization->max[0];
        min = min < existing_min ? min : existing_min;
        max = max > existing_max ? max : existing_max;
      }
    }
    auto quant_params = absl::make_unique<tflite::QuantizationParametersT>();
    quant_params->min.push_back(min);
    quant_params->max.push_back(max);
    subgraph->tensors[tensorid_stat.first]->quantization =
        std::move(quant_params);
  }

  return kTfLiteOk;
}
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

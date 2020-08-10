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
#include "tensorflow/lite/python/optimize/calibration_wrapper.h"

#include <memory>
#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#define TFLITE_PY_CHECK(x)               \
  if ((x) != kTfLiteOk) {                \
    return error_reporter_->exception(); \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()                               \
  if (!interpreter_) {                                                     \
    PyErr_SetString(PyExc_ValueError, "Interpreter was not initialized."); \
    return nullptr;                                                        \
  }

namespace tflite {
namespace calibration_wrapper {

namespace {

using python_utils::PyDecrefDeleter;

std::unique_ptr<tflite::ModelT> CreateMutableModel(const tflite::Model& model) {
  std::unique_ptr<tflite::ModelT> copied_model =
      absl::make_unique<tflite::ModelT>();
  model.UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

inline TensorType TfLiteTypeToSchemaType(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return TensorType_FLOAT32;  // TODO(b/129336260): No schema type for none.
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteFloat16:
      return TensorType_FLOAT16;
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteUInt8:
      return TensorType_UINT8;
    case kTfLiteInt8:
      return TensorType_INT8;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteString:
      return TensorType_STRING;
    case kTfLiteBool:
      return TensorType_BOOL;
    case kTfLiteInt16:
      return TensorType_INT16;
    case kTfLiteComplex64:
      return TensorType_COMPLEX64;
  }
  // No default to get compiler error when new type is introduced.
}

}  // namespace

CalibrationWrapper::CalibrationWrapper(
    std::unique_ptr<tflite::Interpreter> interpreter,
    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver,
    std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
        error_reporter,
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader)
    : interpreter_(std::move(interpreter)),
      error_reporter_(std::move(error_reporter)),
      resolver_(std::move(resolver)),
      model_(std::move(model)),
      reader_(std::move(reader)) {}

CalibrationWrapper::~CalibrationWrapper() {}

PyObject* CalibrationWrapper::Prepare() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::FeedTensor(PyObject* input_value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_value)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input type: expected input to be a list.");
    return nullptr;
  }

  const size_t inputs_size = PyList_Size(input_value);

  if (inputs_size != interpreter_->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input size: expected %ld items got %ld items.",
                 interpreter_->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; i++) {
    PyObject* input = PyList_GetItem(input_value, i);
    if (!input) {
      return nullptr;
    }
    int input_tensor_idx = interpreter_->inputs()[i];
    if (!SetTensor(input_tensor_idx, input)) {
      return nullptr;
    }
  }

  TFLITE_PY_CHECK(interpreter_->Invoke());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::SetTensor(int index, PyObject* value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  const TfLiteTensor* tensor = interpreter_->tensor(index);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor:"
                 " Got tensor of type %s"
                 " but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_SetString(PyExc_ValueError, "Cannot set tensor: Dimension mismatch");
    return nullptr;
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_SetString(PyExc_ValueError,
                      "Cannot set tensor: Dimension mismatch");
      return nullptr;
    }
  }

  size_t size = PyArray_NBYTES(array);
  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float) {
  TfLiteType input_type = python_utils::TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = python_utils::TfLiteTypeFromPyType(output_py_type);
  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PyErr_SetString(PyExc_ValueError,
                    "Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = tflite::optimize::QuantizeModel(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float, error_reporter_.get());
  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

/*static*/ CalibrationWrapper* CalibrationWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data) {
  using tflite::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  ::tflite::python::ImportNumpy();

  if (python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }
  auto resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader;
  auto status = tflite::optimize::calibration::BuildLoggingInterpreter(
      *model, *resolver, &interpreter, &reader);
  if (status != kTfLiteOk) {
    error_reporter->exception();
    return nullptr;
  }

  auto wrapper = new CalibrationWrapper(
      std::move(interpreter), std::move(resolver), std::move(error_reporter),
      std::move(model), std::move(reader));
  return wrapper;
}

}  // namespace calibration_wrapper
}  // namespace tflite

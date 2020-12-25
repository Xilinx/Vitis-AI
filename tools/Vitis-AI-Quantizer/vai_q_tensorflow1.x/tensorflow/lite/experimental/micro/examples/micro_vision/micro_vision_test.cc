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

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/examples/micro_vision/model_settings.h"
#include "tensorflow/lite/experimental/micro/examples/micro_vision/no_person_image_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_vision/person_detect_model_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_vision/person_image_data.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// TODO(rocky): This is too big for many platforms.  Need to implement a more
// efficient memory manager for intermediate tensors.
// TODO(petewarden): Temporarily reduce the size for Arduino builds, so we can
// make sure the continuous-integration builds work.
#ifdef ARDUINO
constexpr int tensor_arena_size = 10 * 1024;
#else   // ARDUINO
const int tensor_arena_size = 300 * 1024;
#endif  // ARDUINO
uint8_t tensor_arena[tensor_arena_size];

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need.
  tflite::ops::micro::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  // Copy an image with a person into the memory area used for the input.
  const uint8_t* person_data = g_person_data;
  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = person_data[i];
  }

  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // Make sure that the expected "Person" score is higher than the other class.
  uint8_t person_score = output->data.uint8[kPersonIndex];
  uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  error_reporter->Report(
      "person data.  person score: %d, no person score: %d\n", person_score,
      no_person_score);
  TF_LITE_MICRO_EXPECT_GT(person_score, no_person_score);

  // Now test with a different input, from an image without a person.
  const uint8_t* no_person_data = g_no_person_data;
  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = no_person_data[i];
  }

  // Run the model on this "No Person" input.
  invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(3, output->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // Make sure that the expected "No Person" score is higher.
  person_score = output->data.uint8[kPersonIndex];
  no_person_score = output->data.uint8[kNotAPersonIndex];
  error_reporter->Report(
      "no person data.  person score: %d, no person score: %d\n", person_score,
      no_person_score);
  TF_LITE_MICRO_EXPECT_GT(no_person_score, person_score);

  error_reporter->Report("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END

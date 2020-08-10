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

// Template sketch that calls into the detailed TensorFlow Lite example code.

// Include an empty header so that Arduino knows to build the TF Lite library.
#include <TensorFlowLite.h>

// TensorFlow Lite defines its own main function
extern int tflite_micro_main(int argc, char* argv[]);

// So the example works with or without a serial connection,
// wait to see one for 5 seconds before giving up.
void waitForSerial() {
  int start = millis();
  while(!Serial) {
    int diff = millis() - start;
    if (diff > 5000) break;
  }
}

// Runs once when the program starts
void setup() {
  waitForSerial();
  tflite_micro_main(0, NULL);
}

// Leave the loop unused
void loop() {
}
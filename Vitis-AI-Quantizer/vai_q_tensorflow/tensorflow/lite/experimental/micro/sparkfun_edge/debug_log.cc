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

// Implementation for the DebugLog() function that prints to the UART on the
// SparkFun Edge microcontroller. The same should work for other targets using
// the Ambiq Apollo 3.

#include "tensorflow/lite/experimental/micro/debug_log.h"

#include "am_bsp.h"   // NOLINT
#include "am_util.h"  // NOLINT

extern "C" void DebugLog(const char* s) {
  static bool is_initialized = false;
  if (!is_initialized) {
    am_bsp_uart_printf_enable();
    is_initialized = true;
  }

  am_util_stdio_printf("%s", s);
}


/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "vitis/ai/general.hpp"

using namespace std;
using namespace google::protobuf;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cout << "usage: generate_proto <model> <img_file> <output_file>" << endl;
    return 1;
  }

  auto image = cv::imread(argv[2]);
  if (image.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto model = vitis::ai::General::create(argv[1], true);
  if (model) {
    auto result = model->run(image);
    cout << "result = " << result.DebugString() << endl;
    fstream output(argv[3], ios::out | ios::trunc | ios::binary);
    if (!result.SerializeToOstream(&output)) {
      cerr << "failed to write result to output.bin." << endl;
      return 1;
    }
    cout << "success to write result to output.bin." << endl;
  } else {
    cerr << "no such model, ls -l /usr/share/vitis-ai-library to see available "
            "models."
         << endl;
  }

  return 0;
}

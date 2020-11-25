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

#include <google/protobuf/text_format.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "vitis/ai/general.hpp"
using namespace std;

std::string g_model_name = "resnet50";
std::string g_image_file = "";

static void usage() {
  std::cout << "usage: test_general <model_name> <img_file> " << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    usage();
    return 1;
  }
  g_model_name = argv[1];
  g_image_file = argv[2];
  auto image = cv::imread(g_image_file);
  if (image.empty()) {
    std::cerr << "cannot load " << g_image_file << std::endl;
    abort();
  }
  auto model = vitis::ai::General::create(g_model_name, true);
  if (model) {
    auto result = model->run(image);
    cerr << "result = " << result.DebugString() << endl;
  } else {
    cerr << "no such model, ls -l /usr/share/vitis-ai-library to see available "
            "models"
         << endl;
  }
  return 0;
}

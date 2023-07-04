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

#include <sstream>
#include "pointpillars_onnx.hpp"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name> " << "  <bin_url> [ <bin_url> ]..." << std::endl;
    abort();
  }

  auto det = OnnxPointpillars::create(argv[1]);

  V2F PointCloud(argc-2);
  std::vector<float*> vf(argc-2);
  V1I vi(argc-2);

  for(int i=2; i<argc; i++) {
    int len = getfloatfilelen(argv[i]);
    PointCloud[i-2].resize(len);
    myreadfile(PointCloud[i-2].data(), len, argv[i]);
    vf[i-2] = PointCloud[i-2].data();
    vi[i-2] = len;
  }

  auto res = det->run(vf, vi);

  // print result
  for(unsigned int k=0; k<res.size(); k++)  {
    std::cout <<"\nbatch " << k <<  "\n";
    for (unsigned int i = 0; i < res[k].ppresult.final_box_preds.size(); i++) {
      std::cout << res[k].ppresult.label_preds[i] << "     " << std::fixed
                << std::setw(11) << std::setprecision(6) << std::setfill(' ')
                << res[k].ppresult.final_box_preds[i][0] << " "
                << res[k].ppresult.final_box_preds[i][1] << " "
                << res[k].ppresult.final_box_preds[i][2] << " "
                << res[k].ppresult.final_box_preds[i][3] << " "
                << res[k].ppresult.final_box_preds[i][4] << " "
                << res[k].ppresult.final_box_preds[i][5] << " "
                << res[k].ppresult.final_box_preds[i][6] << "     "
                << res[k].ppresult.final_scores[i] << "\n";
    }
  }
  return 0;
}


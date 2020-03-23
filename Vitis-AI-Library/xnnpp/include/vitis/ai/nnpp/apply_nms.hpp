/*
 * Copyright 2019 Xilinx Inc.
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

#include <vector>
using namespace std;
  /**
   * @brief Function to do the nms algorithm
   *
   * @param boxes.  vector of coordinates of boxes. 
   *                the vector<float> is coordinate: {x, y, w, h}
   * @param scores. vector of scores.  
   * @param nms.  the nms threshhold.
   * @param conf. confidence to filter the selected result.
   *
   * @return The vector holding returned selected position after nms
   */
void applyNMS(const vector<vector<float>> &boxes, const vector<float> &scores,
              const float nms, const float conf, vector<size_t> &res); 

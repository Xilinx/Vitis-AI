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
namespace vitis {
namespace ai {
class TimeMeasure {
 public:
  TimeMeasure();
  ~TimeMeasure();
  TimeMeasure& operator=(const TimeMeasure& rhs) = delete;

 public:
  static TimeMeasure& getThreadLocalForDpu();

 public:
  void reset();
  void add(int time);
  int get();

  int consuming_time_;
};
}  // namespace ai
}  // namespace vitis

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
#pragma once
#include <map>
#include <mutex>
#include <vector>

using std::map;
using std::vector;

namespace xilinx {
namespace ai {

class StateMap {
 public:
  enum State {
    INIT = 0,
    DET_ST = 1,
    DET_ED = 2,
    DET_TO = 3,
    TRC_ST = 4,
    TRC_ED = 5
  };

  StateMap();
  ~StateMap();

  bool add(int id, State new_state);

  bool set(int id, State new_state);
  void setCur(State new_state);

  State get(int id);
  int getCurId();
  State getCurState();

  void clearState(State target_state);
  void clearBadStates();
  void clearAll();

  void addBadState(State bad_state);

  void print();

  bool updateLastTrackedId(int id);
  int getLastTrackedId();

 private:
  void updateCur();

  map<int, State> *m_data_;

  int cur_id_;
  int last_tracked_id_;
  State cur_state_;
  vector<State> bad_states_;
  mutable std::mutex mtx_;
};

}  // namespace ai
}  // namespace xilinx

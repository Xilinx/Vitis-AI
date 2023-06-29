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

#include "state_map.hpp"
#include <algorithm>
#include "common.hpp"

namespace vitis {
namespace ai {

StateMap::StateMap()
    : m_data_(new map<int, State>),
      cur_id_(-1),
      last_tracked_id_(-1),
      cur_state_(State(0)) {}

StateMap::~StateMap() { delete m_data_; }

StateMap::State StateMap::get(int id) { return m_data_->at(id); }

int StateMap::getCurId() {
  std::lock_guard<std::mutex> lock(this->mtx_);
  return cur_id_;
}

StateMap::State StateMap::getCurState() {
  std::lock_guard<std::mutex> lock(this->mtx_);
  return cur_state_;
}

void StateMap::updateCur() {
  DLOG(INFO) << "try updateCur";
  auto it = m_data_->begin();
  while (it != m_data_->end() &&
         std::find(bad_states_.begin(), bad_states_.end(), it->second) !=
             bad_states_.end()) {
    it++;
  }
  if (it != m_data_->end()) {
    DLOG(INFO) << "updateCur: " << it->first;
    cur_id_ = it->first;
    cur_state_ = it->second;
  } else {
    cur_id_ = -1;
    cur_state_ = State(0);
  }
}

bool StateMap::set(int id, State new_state) {
  std::lock_guard<std::mutex> lock(this->mtx_);
  DLOG(INFO) << "try set: " << id << " state: " << new_state;
  bool success = false;
  if (m_data_->find(id) != m_data_->end()) {
    m_data_->at(id) = new_state;
    updateCur();
    success = true;
  }
  return success;
}

bool StateMap::updateLastTrackedId(int id) {
  std::lock_guard<std::mutex> lock(this->mtx_);
  DLOG(INFO) << "try setLastLocked " << id;
  if (last_tracked_id_ != -1 && id < last_tracked_id_) {
    LOG(WARNING) << "new last_tracked_id must be greater than the current one.";
    return false;
  } else {
    last_tracked_id_ = id;
  }
  return true;
}

int StateMap::getLastTrackedId() {
  std::lock_guard<std::mutex> lock(this->mtx_);
  return last_tracked_id_;
}

void StateMap::addBadState(State bad_state) {
  std::lock_guard<std::mutex> lock(this->mtx_);
  bad_states_.emplace_back(bad_state);
}

bool StateMap::add(int id, State new_state) {
  std::lock_guard<std::mutex> lock(this->mtx_);
  DLOG(INFO) << "try add " << id << " state: " << new_state;
  if (id <= last_tracked_id_) {
    DLOG(WARNING) << "new id must be greater than last tracked_id, new id: "
                  << id << " vs last_tracked_id: " << last_tracked_id_;
    return false;
  }
  bool success = false;
  if (m_data_->find(id) != m_data_->end()) {
    DLOG(WARNING) << "fail to add new id to exist id: " << id;
  } else {
    (*m_data_)[id] = new_state;
    updateCur();
    success = true;
  }
  return success;
}

void StateMap::clearState(State target_state) {
  DLOG(INFO) << "in clearState  state: " << target_state;
  for (auto it = m_data_->begin(); it != m_data_->end();) {
    if (it->second == target_state) {
      m_data_->erase(it++);
    } else {
      it++;
    }
  }
  last_tracked_id_ = -1;
  cur_id_ = -1;
  cur_state_ = State(0);
}

void StateMap::clearAll() {
  m_data_->clear();

  cur_id_ = -1;
  last_tracked_id_ = -1;
  cur_state_ = State(0);
}

void StateMap::clearBadStates() {
  std::lock_guard<std::mutex> lock(this->mtx_);
  // DLOG(INFO) << "try clearBadState";
  for (auto bad_state : bad_states_) {
    // DLOG(INFO) << "try clearBadStates: " << bad_state;
    clearState(bad_state);
  }
}

void StateMap::print() {
  std::lock_guard<std::mutex> lock(this->mtx_);
  LOG(INFO) << std::string(50, '+');
  for (auto& it : *m_data_) {
    LOG(INFO) << "StateMap[" << it.first << "]= " << it.second;
  }
  LOG(INFO) << "cur_id_ = " << cur_id_;
  LOG(INFO) << std::string(50, '-');
}

}  // namespace ai
}  // namespace vitis

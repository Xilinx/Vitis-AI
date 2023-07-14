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

#include "./buffer_object_view.hpp"

#include <glog/logging.h>

#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_BUFFER_OBJECT, "0")

namespace xir {

BufferObjectView::BufferObjectView(BufferObject* parent, size_t offset,
                                   size_t size)
    : BufferObject(),   //
      parent_{parent},  //
      offset_{offset},  //
      size_{size}       //
{}

BufferObjectView::~BufferObjectView() {  //
}

const void* BufferObjectView::data_r() const {  //
  return reinterpret_cast<const char*>(parent_->data_r()) + offset_;
}

void* BufferObjectView::data_w() {  //
  return reinterpret_cast<char*>(parent_->data_w()) + offset_;
}

size_t BufferObjectView::size() { return size_; }

uint64_t BufferObjectView::phy(size_t offset) {
  return parent_->phy(offset_ + offset);
}

void BufferObjectView::sync_for_read(uint64_t offset, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT)) << "sync_for_read "            //
                                               << "offset " << offset << " "  //
                                               << "size " << size << " "      //
                                               << std::endl;
  CHECK_LE(offset + size, size_);
  parent_->sync_for_read(offset_ + offset, size);
}

void BufferObjectView::sync_for_write(uint64_t offset, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT)) << "sync_for_write "           //
                                               << "offset " << offset << " "  //
                                               << "size " << size << " "      //
                                               << std::endl;
  CHECK_LE(offset + size, size_);
  parent_->sync_for_write(offset_ + offset, size);
}

/// copy from host to this buffer object
void BufferObjectView::copy_from_host(const void* buf, size_t size,
                                      size_t offset) {
  parent_->copy_from_host(buf, size, offset_ + offset);
}

/// copy from this buffer object to host
void BufferObjectView::copy_to_host(void* buf, size_t size, size_t offset) {
  parent_->copy_to_host(buf, size, offset_ + offset);
}

}  // namespace xir
DECLARE_INJECTION(xir::BufferObject, xir::BufferObjectView,
                  xir::BufferObject*&&, size_t&&, size_t&&);

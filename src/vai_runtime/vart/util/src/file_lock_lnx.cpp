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
#include "./file_lock.hpp"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace vitis {
namespace ai {

static int obtain_fd(const char* filename) {
  auto fd = open(filename, FD_CLOEXEC | O_WRONLY | O_CREAT, 0777);
  CHECK_GE(fd, 0) << "cannot open file: " << filename;
  if (fchmod(fd, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)) {
    LOG_IF(ERROR, errno != EPERM)
        << "cannot chmod file: " << filename << ", errno = " << errno;
  }
  return fd;
}

FileLock::FileLock(const std::string filename, size_t offset)
    : fd_(obtain_fd(filename.c_str())), offset_(offset) {}

FileLock::~FileLock() { close(fd_); }

void FileLock::lock() {
  auto ret = lockf(fd_, F_LOCK, offset_);
  CHECK_EQ(ret, 0) << "cannot lockf: " << errno;
  return;
}

bool FileLock ::try_lock() {
  auto ret = true;
  int r = 0;
  r = lockf(fd_, F_TLOCK, offset_);
  if (r == 0) {
    ret = true;
  } else if (errno == EAGAIN || errno == EACCES) {
    ret = false;
  } else {
    LOG(FATAL) << "unknown lock error: " << errno;
  }
  return ret;
}

void FileLock::unlock() {
  auto ret = lockf(fd_, F_ULOCK, offset_);
  CHECK_EQ(ret, 0) << "unlock failed";
  return;
}

}  // namespace ai
}  // namespace vitis

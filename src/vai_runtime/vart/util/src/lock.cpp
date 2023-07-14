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

#include "vitis/ai/lock.hpp"

#include <glog/logging.h>
#include <locale>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

#include "./file_lock.hpp"
#include "vitis/ai/env_config.hpp"

#ifdef ENABLE_REDIS
#include "./redis_lock.hpp"

DEF_ENV_PARAM_2(XLNX_REDIS_SERVER, "127.0.0.1", std::string);
DEF_ENV_PARAM_2(XLNX_REDIS_PORT, "6379", size_t);
#endif

DEF_ENV_PARAM(DEBUG_LOCK, "0");
DEF_ENV_PARAM_2(XLNX_LOCK_TYPE, "redis", std::string);

namespace vitis {
namespace ai {

Lock::Lock(){};
Lock::~Lock(){};

#ifdef ENABLE_REDIS
static bool is_redis_server_alive() {
  auto ctx = std::unique_ptr<redisContext>(redisConnect(
      ENV_PARAM(XLNX_REDIS_SERVER).c_str(), ENV_PARAM(XLNX_REDIS_PORT)));
  if (ctx.get() == nullptr || ctx->err) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK)) << "Can't connect to redis server.";
    return false;
  }
  return true;
}
#endif

std::unique_ptr<Lock> Lock::create(const std::string& lock_name_input) {
  // make sure there is no whitespace in lock_name
  std::string lock_name = lock_name_input;
  lock_name.erase(std::remove_if(lock_name.begin(), lock_name.end(),
                                 [](char& c) {
                                   return std::isspace<char>(
                                       c, std::locale::classic());
                                 }),
                  lock_name.end());

#ifdef ENABLE_REDIS
  if (ENV_PARAM(XLNX_LOCK_TYPE) == std::string("redis") &&
      is_redis_server_alive()) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK)) << "Create redis lock";
    return std::unique_ptr<Lock>(new RedisLock(lock_name));
  }
#endif

  LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK)) << "Create file lock";
  return std::unique_ptr<Lock>(new FileLock("/tmp/" + lock_name));
}

}  // namespace ai
}  // namespace vitis

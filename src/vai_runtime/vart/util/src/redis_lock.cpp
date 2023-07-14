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

#include "./redis_lock.hpp"

#include <glog/logging.h>
#include <unistd.h>

#include <boost/mpl/assert.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <type_traits>

#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_LOCK, "0");
DEF_ENV_PARAM_2(XLNX_REDIS_SERVER, "127.0.0.1", std::string);
DEF_ENV_PARAM_2(XLNX_REDIS_PORT, "6379", size_t);
DEF_ENV_PARAM_2(XLNX_REDIS_TIMEOUT_S, "1", size_t);

namespace vitis {
namespace ai {

static std::string generate_token() {
  std::ostringstream pid;
  pid << getpid();
  auto timestamp = std::chrono::high_resolution_clock::now();
  srand(time(0));
  auto token = pid.str() + "_" +
               std::to_string(timestamp.time_since_epoch().count()) + "_" +
               std::to_string(rand());
  LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK)) << "token is " << token;
  return token;
}

static std::unique_ptr<redisContext> get_ctx() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK))
      << "redis server: " << ENV_PARAM(XLNX_REDIS_SERVER)
      << "  port: " << ENV_PARAM(XLNX_REDIS_PORT);

  auto ctx = std::unique_ptr<redisContext>(redisConnect(
      ENV_PARAM(XLNX_REDIS_SERVER).c_str(), ENV_PARAM(XLNX_REDIS_PORT)));
  LOG_IF(FATAL, !ctx) << "Couldn't create redisContext";
  LOG_IF(FATAL, ctx->err) << "Redis connect error: " << ctx->errstr;
  return ctx;
}

RedisLock::RedisLock(const std::string& device_name)
    : ctx_(get_ctx()),
      device_name_(device_name),
      token_(generate_token()),
      timeout_(ENV_PARAM(XLNX_REDIS_TIMEOUT_S)),
      is_finished_(false),
      mutex_() {}

RedisLock::~RedisLock() { heart_beat_thread_.join(); }

bool RedisLock::is_own_locked() {
  std::lock_guard<std::mutex> redis_cmd_lock(mutex_);
  auto reply = static_cast<redisReply*>(
      redisCommand(ctx_.get(), "GET %s", device_name_.c_str()));
  CHECK_NOTNULL(reply);
  CHECK(!ctx_->err) << "redis command error: " << ctx_->err;
  if (reply->str && std::string(reply->str) == token_) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK))
        << device_name_ << " has already locked by current thread";
    freeReplyObject(reply);
    return true;
  }
  freeReplyObject(reply);
  return false;
}

bool RedisLock::try_lock() {
  if (is_own_locked()) {
    return true;
  }

  std::lock_guard<std::mutex> redis_cmd_lock(mutex_);

  auto reply = static_cast<redisReply*>(
      redisCommand(ctx_.get(), "SET %s %s EX %s NX", device_name_.c_str(),
                   token_.c_str(), std::to_string(timeout_ + 1).c_str()));
  CHECK_NOTNULL(reply);
  CHECK(!ctx_->err) << "redis command error: " << ctx_->err;

  if (reply->str && std::strcmp(reply->str, "OK") == 0) {
    freeReplyObject(reply);
    if (is_finished_) {
      // It should wait heart beat thread stop if try lock again.
      heart_beat_thread_.join();
      is_finished_ = false;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK)) << "start heart beat thread";
    heart_beat_thread_ = std::thread([this]() { heart_beat(); });
    return true;
  }

  freeReplyObject(reply);
  return false;
}

void RedisLock::lock() {
  while (!try_lock()) {
    sleep(1);
  };
}

void RedisLock::unlock() {
  is_finished_ = true;

  const std::string cmd =
      "if redis.call('get', KEYS[1]) ~= ARGV[1] then \
         return 0; \
      else \
         redis.call('del', KEYS[1]); \
         return 1; \
      end";

  std::lock_guard<std::mutex> redis_cmd_lock(mutex_);

  auto reply = static_cast<redisReply*>(
      redisCommand(ctx_.get(), "EVAL %s 1 %s %s", cmd.c_str(),
                   device_name_.c_str(), token_.c_str()));
  CHECK_NOTNULL(reply);
  CHECK(!ctx_->err) << "redis command error: " << ctx_->errstr;
  LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK))
      << "unlock lua script return value: " << reply->integer;
  freeReplyObject(reply);
}

void RedisLock::heart_beat() {
  if (!is_own_locked()) {
    LOG_IF(FATAL, !is_finished_)
        << "Heart beat thread is running while the device is not "
           "locked by current process";
  }

  while (!is_finished_) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_LOCK)) << "heart beat......................";
    {
      std::lock_guard<std::mutex> redis_cmd_lock(mutex_);
      auto reply = static_cast<redisReply*>(
          redisCommand(ctx_.get(), "EXPIRE %s %s", device_name_.c_str(),
                       std::to_string(timeout_).c_str()));
      CHECK_NOTNULL(reply);
      CHECK(!ctx_->err) << "redis command error: " << ctx_->err;
      freeReplyObject(reply);
    }
    usleep(timeout_ * 1000000 - 200000);
  }
}

}  // namespace ai
}  // namespace vitis

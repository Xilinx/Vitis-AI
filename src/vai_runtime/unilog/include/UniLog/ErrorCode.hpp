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

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "UniLogExport.hpp"

// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;

class ErrorCodeFactory;
class ErrorCode {
public:
  ErrorCode() = delete;
  UNILOG_DLLESPEC ErrorCode(const std::string &errID,
                            const std::string &errDsp = "",
                            const std::string &errDebugInfo = "")
      : errID_(errID), errDsp_(errDsp), _errDebugInfo_(errDebugInfo){};
  UNILOG_DLLESPEC ErrorCode(const ErrorCode &errCode);

  UNILOG_DLLESPEC const std::string &getErrID() const { return this->errID_; }
  UNILOG_DLLESPEC const std::string &getErrDsp() const { return this->errDsp_; }
  UNILOG_DLLESPEC const std::string &getErrDebugInfo() const;
  UNILOG_DLLESPEC const std::string &getErrMsg() const {
    return this->errMsg_;
  } // non-violated read
  UNILOG_DLLESPEC std::string
  extractErrMsg(); // violated read, it will destroy after read

  UNILOG_DLLESPEC ErrorCode &operator<<(const std::string &errMsg);

private:
  std::string errID_;
  std::string errDsp_;
  std::string _errDebugInfo_;
  std::string errMsg_;
};

UNILOG_DLLESPEC
std::ostream &operator<<(std::ostream &os, ErrorCode &errCode);

class UNILOG_DLLESPEC ErrorCodeFactory {
public:
  static ErrorCodeFactory &Instance();
  void registerErrorCode(const std::string &errID,
                         std::shared_ptr<ErrorCode> errCodePtr);
  ErrorCode &genErrorCode(const std::string &errID);
  void
  dumpErrorCodeMap(const std::string &errMapFileName = "_ERRORCODE_MAP.csv");

private:
  ErrorCodeFactory() {}
};

class UNILOG_DLLESPEC ErrorCodeRegister {
public:
  ErrorCodeRegister() = delete;
  ErrorCodeRegister(const std::string &errID, const std::string &errDsp = "",
                    const std::string &errDebugInfo = "");
};

#define REGISTER_ERROR_CODE(ERRID, ERRDSP, ERRDEBUGINFO)                       \
  static const ErrorCodeRegister ERRID##OBJ_(#ERRID, ERRDSP, ERRDEBUGINFO)

#define GEN_ERROR(ERRID) ErrorCodeFactory::Instance().genErrorCode(#ERRID)

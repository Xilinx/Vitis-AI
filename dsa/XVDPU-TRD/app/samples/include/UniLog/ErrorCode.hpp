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
using namespace std;

class ErrorCodeFactory;
class ErrorCode {
public:
  ErrorCode() = delete;
  ErrorCode(const string &errID, const string &errDsp = "",
            const string &errDebugInfo = "")
      : errID_(errID), errDsp_(errDsp), _errDebugInfo_(errDebugInfo){};
  ErrorCode(const ErrorCode &errCode);

  const string &getErrID() const { return this->errID_; }
  const string &getErrDsp() const { return this->errDsp_; }
  const string &getErrDebugInfo() const;
  const string &getErrMsg() const { return this->errMsg_; } // non-violated read
  string extractErrMsg(); // violated read, it will destroy after read

  ErrorCode &operator<<(const string &errMsg);

private:
  string errID_;
  string errDsp_;
  string _errDebugInfo_;
  string errMsg_;
};

ostream &operator<<(ostream &os, ErrorCode &errCode);

class ErrorCodeFactory {
public:
  static ErrorCodeFactory &Instance() {
    static ErrorCodeFactory instanceErrorCodeFactory{};
    return instanceErrorCodeFactory;
  }
  void registerErrorCode(const string &errID, shared_ptr<ErrorCode> errCodePtr);
  ErrorCode &genErrorCode(const string &errID);
  void dumpErrorCodeMap(const string &errMapFileName = "_ERRORCODE_MAP.csv");

private:
  ErrorCodeFactory() {}

private:
  map<string, shared_ptr<ErrorCode>> mapErrorCode_;
};

class ErrorCodeRegister {
public:
  ErrorCodeRegister() = delete;
  ErrorCodeRegister(const string &errID, const string &errDsp = "",
                    const string &errDebugInfo = "");
};

#define REGISTER_ERROR_CODE(ERRID, ERRDSP, ERRDEBUGINFO)                       \
  static const ErrorCodeRegister ERRID##OBJ_(#ERRID, ERRDSP, ERRDEBUGINFO)

#define GEN_ERROR(ERRID) ErrorCodeFactory::Instance().genErrorCode(#ERRID)

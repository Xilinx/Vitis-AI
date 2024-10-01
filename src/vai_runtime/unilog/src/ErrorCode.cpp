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

#include "UniLog/ErrorCode.hpp"

#include "ErrorCode.imp"
#include "UniLog/UniLog.hpp"
using std::cerr;
using std::cout;
using std::endl;
using std::ios;
using std::make_shared;
using std::ofstream;
using std::shared_ptr;
using std::string;

ErrorCode::ErrorCode(const ErrorCode &errCode) {
  errID_ = errCode.errID_;
  errDsp_ = errCode.errDsp_;
  _errDebugInfo_ = errCode._errDebugInfo_;
  errMsg_ = errCode.errMsg_;
}

const string &ErrorCode::getErrDebugInfo() const { return _errDebugInfo_; }

string ErrorCode::extractErrMsg() {
  string errMsg{};
  this->errMsg_.swap(errMsg);
  return errMsg;
}

ErrorCode &ErrorCode::operator<<(const string &errMsg) {
  errMsg_ += errMsg;
  return *this;
}

std::ostream &operator<<(std::ostream &os, ErrorCode &errCode) {
  os << "[" << errCode.getErrID() << "]"
     << "[" << errCode.getErrDsp() << "]";
#if defined(UNI_LOG_DEBUG_MODE)
  os << "[" << errCode.getErrDebugInfo() << "]";
#endif
  os << "[" << errCode.getErrMsg() << "]";
  return os;
}

ErrorCodeFactory &ErrorCodeFactory::Instance() {
  static ErrorCodeFactory instanceErrorCodeFactory{};
  return instanceErrorCodeFactory;
}

static std::map<std::string, std::shared_ptr<ErrorCode>> &mapErrorCode() {
  static std::map<std::string, std::shared_ptr<ErrorCode>> mapErrorCode_;
  return mapErrorCode_;
};

void ErrorCodeFactory::registerErrorCode(
    const string &errID, const shared_ptr<ErrorCode> errCodePtr) {
#if UNILOG_USE_DLL == 0
  // when linking with static lib, error code might not be
  // registered, because we cannot control the initialization order.
  // so that a error code might be registered more than twice.
#else
  auto keyValue = mapErrorCode().find(errID);
  if (keyValue != mapErrorCode().end()) {
    cerr << "Error ID (" << errID << ") has been registered, please modify!!!"
         << endl;
    abort();
  }
#endif
  mapErrorCode()[errID] = errCodePtr;
}

ErrorCode &ErrorCodeFactory::genErrorCode(const string &errID) {
  auto generator = mapErrorCode().find(errID);
  if (generator == mapErrorCode().end()) {
#if UNILOG_USE_DLL == 0
    // when linking with static lib, error code might not be
    // registered, because we cannot control the initialization order.
    mapErrorCode()[errID] = std::make_shared<ErrorCode>(errID);
#else
    cerr << errID << " is not registered, please register before using it!"
         << endl;
    abort();
#endif
  }
  generator = mapErrorCode().find(errID);
  if (generator == mapErrorCode().end()) {
    cerr << "never goes here";
    abort();
  }
  auto ret = generator->second;
  return *ret;
}

void ErrorCodeFactory::dumpErrorCodeMap(const string &errMapFileName) {
#if defined(UNI_LOG_DEBUG_MODE)
  ofstream dumpfile;
  dumpfile.open(errMapFileName, ios::out | ios::trunc);
  dumpfile << "Error ID, Error Description, Error Debug Info" << endl;
  for (auto iter : mapErrorCode()) {
    auto errorcode = iter.second;
    dumpfile << errorcode->getErrID() << ",\"" << errorcode->getErrDsp()
             << "\",\"" << errorcode->getErrDebugInfo() << "\"" << endl;
  }
  dumpfile.close();
#endif
}

ErrorCodeRegister::ErrorCodeRegister(const string &errID, const string &errDsp,
                                     const string &errDebugInfo) {
#if defined(UNI_LOG_DEBUG_MODE)
  ErrorCodeFactory::Instance().registerErrorCode(
      errID, make_shared<ErrorCode>(errID, errDsp, errDebugInfo));
#else
  ErrorCodeFactory::Instance().registerErrorCode(
      errID, std::make_shared<ErrorCode>(errID, errDsp));
#endif
}

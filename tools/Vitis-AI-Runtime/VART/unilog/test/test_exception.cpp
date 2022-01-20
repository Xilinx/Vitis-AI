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
#include "UniLog/UniLog.hpp"
#include <exception>
#include <iostream>

using namespace std;

void test_error(int n) {
  if (n > 0) {
    test_error(n - 1);
  } else {
    throw 0;
  }
}

REGISTER_ERROR_CODE(M_ERROR, "No problem!", "NO NO PROBLEM");
void test_ErrorCode() {
  // ErrorBase error{"CORE_DUMP", 0x123, "abc", "ABC"};
  // throw error;
  // throw * ErrorCodeFactory::Instance().getErrorCode("CORE_DUMP");
  // throw 8;
  throw GEN_ERROR(M_ERROR) << "There is throw an error, "
                           << "Please handle it!";
}

int main(int argc, char *argv[]) {
  try {
    test_error(10);
  } catch (int errorCode) {
    cout << "The Error Code is " << errorCode << "." << endl;
  }
  cout << "The Error Has been handled!" << endl;

  try {
    test_ErrorCode();
  } catch (ErrorCode e) {
    cout << "The Error ID is " << e.getErrID() << endl;
    cout << "The display is " << e.getErrDsp() << endl;
    cout << "The debug info is " << e.getErrDebugInfo() << endl;
    auto ecopy = e;
    cout << "The message is " << e.getErrMsg() << endl;
    cout << "Test the << stream output:" << e << endl;
    cout << "The message is " << e.extractErrMsg() << endl;
    cout << "Test the << stream output(original e):" << e << endl;
    cout << "Test the << stream output(copy of e):" << ecopy << endl;
  }
  ErrorCodeFactory::Instance().dumpErrorCodeMap();

  return 0;
}

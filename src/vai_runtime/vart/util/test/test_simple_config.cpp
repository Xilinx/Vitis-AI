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
#include <iostream>
#include <memory>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/simple_config.hpp>
using namespace vitis::ai;

int main(int argc, char* argv[]) {
  auto config = SimpleConfig::getOrCreateSimpleConfig(argv[1]);
  if (!config) {
    std::cout << "cannot find file! file=" << argv[1] << std::endl;
    return 0;
  }
  for (auto idx = 2; idx < argc; ++idx) {
    std::cout << "key: " << argv[idx] << "; value="
              << (*config)(std::string(argv[idx])).as<std::string>()
              << std::endl;
  }
  __TIC__(simple_config)
  int hello_abc_x = (*config)("hello")("abc")("x").as<int>();
  std::cout << "hello_abc_x " << hello_abc_x << " "  //
            << std::endl;
  std::string hi_abc = (*config)("hi")("abc").as<std::string>();
  std::cout << "hi_abc " << hi_abc << " "  //
            << std::endl;

  std::string ll_0_abc = (*config)("ll")(0)("abc").as<std::string>();
  std::cout << "ll_0_abc " << ll_0_abc << " " << std::endl;

  std::string ll_0_abc_2 = (*config)("ll")[0]("abc").as<std::string>();
  std::cout << "ll_0_abc_2 " << ll_0_abc_2 << " " << std::endl;

  float f_abc_x = (*config)("f")("abc")("x").as<float>();
  std::cout << "f_abc_x " << f_abc_x << " "  //
            << std::endl;
  std::cout << "has hello: " << (*config).has("hello")
            << std::endl;  // ("abc")("x")
  std::cout << "has hello.abc: " << (*config)("hello").has("abc")
            << std::endl;  // ("abc")("x")
  std::cout << "has hello.abc123: " << (*config)("hello").has("abc123")
            << std::endl;  // ("abc")("x")
  __TOC__(simple_config)
  return 0;
}

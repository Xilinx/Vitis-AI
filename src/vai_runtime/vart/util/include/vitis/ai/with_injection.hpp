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
#pragma once
#include <cstdlib>
#include <functional>
#include <iostream>  // it is not necessary, because MSVC strange optimization behaviour, see counter.
#include <memory>
#include <string>
#include <type_traits>
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wundefined-var-template"
#endif
namespace vitis {
namespace ai {
// it is not safe to use template acorss DLLs, so this class is not annotated
// with VART_UTIL_DLLSPEC.
template <typename T>
struct WithInjection {
 public:
  template <typename... Args>
  using factory_method_t =
      typename std::add_pointer<std::unique_ptr<T>(Args&&...)>::type;
  template <typename... Args>
  using priority_t = int;
  // make it easier to access this class, e.g. T::with_injection_t
  using with_injection_t = WithInjection<T>;
  // it is not safe to call template function across DLL, so that every module
  // should create factory method explicitly, see test/device_scheduler.cpp for
  // example.
  template <typename... Args>
  static std::unique_ptr<T> create0(Args&&... args) {
    if (the_factory_method<Args&&...> == nullptr) {
      std::cerr << "the factory method is empty!" << std::endl;
      std::abort();
    }
    auto ret = the_factory_method<Args&&...>(std::forward<Args>(args)...);
    dynamic_cast<WithInjection<T>*>(ret.get())->initialize();
    return ret;
  }

 public:
  template <typename Subclass>
  struct factory_method_generator_t {
    template <typename... Args>
    static constexpr factory_method_t<Args&&...> generate() {
      // counter is useless. I have to use std::cout to create some sort of side
      // effect, otherwise, MSVC 2017 initialized the factory_method with
      // nullptr for no reason.
      // for example, without following if statement, test_injector with trigger
      // the above std::abort "factory metheod is empty", i.e.
      // the_factory_method == nullptr.
      counter = counter + 1;
      if (counter < 1) {
        std::cout << "hello" << std::endl;
      }
      return [](Args&&... args) -> std::unique_ptr<T> {
        return std::make_unique<Subclass>(std::forward<Args>(args)...);
      };
    }
  };

 private:
  virtual void initialize() {}

 public:
  template <typename... Args>
  static factory_method_t<Args&&...> the_factory_method;
  template <typename... Args>
  static priority_t<Args&&...> the_factory_method_priority;
#if _MSC_VER > 1920
  static inline int counter = 0;
#else
  // see below trigger compiler bug for vs2019
  static int counter;
#endif
};

#if _MSC_VER > 1920
#else
// trigger a compiler bug
template <typename T>
int WithInjection<T>::counter = 0;
#endif
template <typename T>
template <typename... Args>
int WithInjection<T>::the_factory_method_priority = 0;

// template <typename T>
// int WithInjection<T>::the_factory_method_priority = 0;

// this macro is used to define a factory method which might have
// multiply implentation.
#define DECLARE_INJECTION_NULLPTR(super, ...)                                  \
  namespace vitis {                                                            \
  namespace ai {                                                               \
  template <>                                                                  \
  template <>                                                                  \
  super::factory_method_t<__VA_ARGS__>                                         \
      super::with_injection_t::the_factory_method<__VA_ARGS__> = nullptr;      \
  }                                                                            \
  }

// this macro is used to define a sub implemenation which is the
// single implation.
#define DECLARE_INJECTION(super, imp, ...)                                     \
  namespace vitis {                                                            \
  namespace ai {                                                               \
  template <>                                                                  \
  template <>                                                                  \
  super::factory_method_t<__VA_ARGS__>                                         \
      super::with_injection_t::the_factory_method<__VA_ARGS__> =               \
          super::factory_method_generator_t<imp>::generate<__VA_ARGS__>();     \
  }                                                                            \
  }
//
#define REGISTER_INJECTION_BEGIN(super, priority, imp, ...)                    \
  namespace {                                                                  \
  struct anonymouse_with_injection_register {                                  \
    anonymouse_with_injection_register() {                                     \
      if (super::with_injection_t::the_factory_method_priority<__VA_ARGS__> <  \
          priority) {                                                          \
        if (ok()) {                                                            \
          super::with_injection_t::the_factory_method<__VA_ARGS__> =           \
              super::factory_method_generator_t<imp>::generate<__VA_ARGS__>(); \
          super::with_injection_t::the_factory_method_priority<__VA_ARGS__> =  \
              priority;                                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    static bool ok() {
#define REGISTER_INJECTION_END                                                 \
  }                                                                            \
  }                                                                            \
  g_anonymouse_with_injection_register;                                        \
  }

#define DECLARE_INJECTION_IN_SHARED_LIB_WITH_SYMBOL_NAME(super, imp, name,     \
                                                         ...)                  \
  extern "C" super::factory_method_t<__VA_ARGS__> name;                        \
  super::factory_method_t<__VA_ARGS__> name =                                  \
      super::factory_method_generator_t<imp>::generate<__VA_ARGS__>();

#define DECLARE_INJECTION_IN_SHARED_LIB(super, imp, ...)                       \
  DECLARE_INJECTION_IN_SHARED_LIB_WITH_SYMBOL_NAME(super, imp, factory_method, \
                                                   __VA_ARGS__)

}  // namespace ai
}  // namespace vitis

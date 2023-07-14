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

#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef OPTIONAL
#define RTLD_LAZY 0
#define RTLD_GLOBAL 0
#define RTLD_NOW 0
#endif

#ifdef _WIN32
inline void* dlopen(const char* dllname, int) { return LoadLibrary(dllname); }

inline void dlclose(void* handle) { FreeLibrary(HMODULE(handle)); }

inline const char* dlerror() { return ""; }

inline void* dlsym(void* handle, const char* symbol) {
  return GetProcAddress(HMODULE(handle), symbol);
}

#endif

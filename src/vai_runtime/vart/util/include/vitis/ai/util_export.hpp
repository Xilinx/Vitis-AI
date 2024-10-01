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
#if defined(_WIN32)
#if VART_UTIL_USE_DLL == 1
#ifndef VART_UTIL_DLLSPEC
#ifdef VART_UTIL_EXPORT
#define VART_UTIL_DLLSPEC __declspec(dllexport)
#else
#define VART_UTIL_DLLSPEC __declspec(dllimport)
#endif
#else
#define VART_UTIL_DLLSPEC
#endif
#else
#define VART_UTIL_DLLSPEC __attribute__((visibility("default")))
#endif
#endif

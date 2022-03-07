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
#include "vitis/ai/plugin.hpp"

#include <sstream>
namespace vitis {
namespace ai {

#if _WIN32
static std::wstring s2ws(const std::string& s) {
  int len;
  int slength = (int)s.length() + 1;
  len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
  wchar_t* buf = new wchar_t[len];
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
  std::wstring r(buf);
  delete[] buf;
  return r;
}
plugin_t open_plugin(const std::string& name, scope_t scope) {
  return LoadLibraryW(s2ws(name).c_str());
}
void* plugin_sym(plugin_t plugin, const std::string& name) {
  return GetProcAddress(plugin, name.c_str());
}
std::string plugin_error(plugin_t plugin) {
  std::ostringstream str;
  str << "ERROR CODE: " << GetLastError();
  return str.str();
}
void close_plugin(plugin_t plugin) { FreeLibrary(plugin); }

#else
#include <dlfcn.h>
plugin_t open_plugin(const std::string& name, scope_t scope) {
  auto flag_public = (RTLD_LAZY | RTLD_GLOBAL);
  auto flag_private = (RTLD_LAZY | RTLD_LOCAL);
  return dlopen(name.c_str(),
                scope == scope_t::PUBLIC ? flag_public : flag_private);
}
void* plugin_sym(plugin_t plugin, const std::string& name) {
  dlerror();  // clean up error;
  return dlsym(plugin, name.c_str());
}
std::string plugin_error(plugin_t plugin) {
  std::ostringstream str;
  str << "ERROR CODE: " << dlerror();
  return str.str();
}
void close_plugin(plugin_t plugin) { dlclose(plugin); }

#endif
}  // namespace ai
}  // namespace vitis

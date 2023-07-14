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
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef char* (*CAC_FUNC)();

int main(int argc, char** argv) {
  void* handle;
  char* error;
  CAC_FUNC cac_func = NULL;
  char* so[] = {//"libunilog.so.3",
                //"libtarget-factory.so.3",
                "libxir.so.3",
                "libvart-buffer-object.so.3",
                "libvart-dpu-controller.so.3",
                "libvart-dpu-runner.so.3",
                "libvart-mem-manager.so.3",
                "libvart-runner.so.3",
                "libvart-util.so.3",
                "libvart-xrt-device-handle.so.3"};
  for (int i = 0; i < sizeof(so) / sizeof(so[0]); i++) {
    handle = dlopen(so[i], RTLD_LAZY);
    if (!handle) {
      fprintf(stderr, "Open library %s error: %s\n", so[i], dlerror());
      continue;
    }

    dlerror();

    cac_func = dlsym(handle, "xilinx_version");
    if ((error = dlerror()) != NULL) {
      fprintf(stderr, "Symbol xilinx_version not found in %s: %s\n", so[i],
              error);
      dlclose(handle);
      continue;
    }
    printf("%s: %s\n", so[i], cac_func());

    dlclose(handle);
  }
  exit(EXIT_SUCCESS);
}

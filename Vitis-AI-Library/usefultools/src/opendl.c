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
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>


typedef char* (*CAC_FUNC)();

int main(int argc, char **argv)
{
    if (argc != 2){
        fprintf(stderr, "Usage: ./opendl test.so\n");
        exit(EXIT_FAILURE);
    }


    void *handle;
    char *error;
    CAC_FUNC cac_func = NULL;

    handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    } 

    dlerror();


    cac_func = dlsym(handle, "xilinx_version");
    if ((error = dlerror()) != NULL)  {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE); 
    }
    printf("%s\n", cac_func());

    dlclose(handle);
    exit(EXIT_SUCCESS);
 }

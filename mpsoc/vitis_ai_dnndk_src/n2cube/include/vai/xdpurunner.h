/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XDPURUNNER_H
#define XDPURUNNER_H

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <sstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <streambuf>
#include <unordered_map>
#include <future>
#include <map>
#include <list>
#include <iterator>

#include <pthread.h>
#include <semaphore.h>
#include <future>
#include <map>

#include "dnndk/n2cube.h"
#include "vai/dpu_runner.hpp"

namespace vitis {
namespace ai {
typedef struct _DPUJob_ {
    uint32_t idx;
    uint32_t batch_offset;
    std::unordered_map<std::string, std::vector<const float*> > in;
    std::unordered_map<std::string, std::vector<float*> > out;
} DPUJob;

typedef struct _DPUJT_ {
    DPUJob  *job;
    DPUTask *tsk;
} DPUTJ;

typedef struct _DPUJid_ {
    uint32_t idx; //base idx.
    uint32_t msk; //batchSize max to 32.
    uint32_t cnt;
    pthread_mutex_t wait;
} DPUJid;

struct vaiTensorShape {
    unsigned int height;
    unsigned int width;
    unsigned int channel;
};
struct vaiGraphInfo{
    unsigned int inTensorCnt;
    unsigned int outTensorCnt;
    struct vaiTensorShape *inTensorList;
    struct vaiTensorShape *outTensorList;
};

class XdpuRunner : public DpuRunner {
public:
    explicit XdpuRunner(const std::string &path);
    // TODO explicit XdpuRunner(const xir::Subgraph);
    ~XdpuRunner();

    virtual std::pair<uint32_t, int>
    execute_async(const std::vector<TensorBuffer*> &input,
                const std::vector<TensorBuffer*> &output) override;

    virtual int wait(int jobid, int timeout=-1) override;

    virtual TensorFormat get_tensor_format() override {
      return DpuRunner::TensorFormat::NHWC;
    }

    virtual std::vector<Tensor*> get_input_tensors() override;
    virtual std::vector<Tensor*> get_output_tensors() override;

private:
    const std::string path_;

    std::vector<Tensor*> inputs_;
    std::vector<Tensor*> outputs_;

    pthread_mutex_t  mutWL;
    std::list <DPUJid *>  waitList;

    sem_t            semJob;
    pthread_mutex_t  mutJob;
    std::list <DPUJob *>  lstJob;

    sem_t            semTsk;
    pthread_mutex_t  mutTsk;
    std::list <DPUTask *> lstTsk;

    sem_t            semTskJob;
    pthread_mutex_t  mutTskJob;
    std::list <DPUTJ *>   lstTskJob;

    sem_t            semOUT;
    pthread_mutex_t  mutOUT;
    std::list <DPUTJ *>   lstOUT;

    unsigned int g_jobs_idx = 1;
    DPUKernel* g_kernel = NULL;
    struct vaiGraphInfo *g_graphinfo = NULL;
    int g_graphinfo_iLength=0;
    int g_graphinfo_oLength=0;

    static void *_read(void *t);
    static void *_run(void *t);
    static void *_write(void *t);

    int N_THREAD_READ;
    int N_THREAD_RUN;
    int N_THREAD_WRITE;
    int N_TASK_POOL;
};
}
}

#endif

/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_GRAPH_L3_TASK_HPP_
#define _XF_GRAPH_L3_TASK_HPP_

#include <chrono>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <queue>

#include <assert.h>
#include <bitset>
#include <sstream>
#include <xrm.h>

#include <thread>
#include <future>
#include <unistd.h>
namespace xf {
namespace graph {
namespace L3 {

class openXRM {
   public:
    xrmCuGroupResource** resR;
    char** udfCuGroupName;
    openXRM() {
        ctx = (xrmContext*)xrmCreateContext(XRM_API_VERSION_1);
        bool isRight = !(ctx == NULL);
        assert(isRight);
    };

    //  void freeCuGroup(unsigned int deviceNm){
    //      int ret = 0;
    //      for(int i = 0; i< deviceNm; ++i){
    //          ret = xrmUdfCuGroupUndeclare(ctx, udfCuGroupName[i]);
    //          if (ret == XRM_SUCCESS){
    //              printf("INFO: User defined cu group from same device undeclaration success\n");
    //          }else{
    //              printf("ERROR: User defined cu group from same device undeclaration fail\n");
    //              exit(1);
    //          }
    //          delete[] udfCuGroupName[i];
    //      }
    //      delete[] resR;
    //      delete[] udfCuGroupName;
    //  }

    //  void setUpCuGroup(unsigned int deviceNm, unsigned int cuNm, std::string kernelName, std::string kernelAlias,
    //  unsigned int requestLoad){
    //      resR = new xrmCuGroupResource*[deviceNm];
    //      udfCuGroupName = new char* [deviceNm];
    //     // xrmCuGroupResource* resR[deviceNm];
    //     // char udfCuGroupName[deviceNm][XRM_MAX_NAME_LEN];
    //      xrmUdfCuGroupProperty* udfCuGroupProp[deviceNm];
    //      std::string baseName = "udfCuGroupSameDevice";
    //      xrmUdfCuListProperty* udfCuListProp;
    //      xrmUdfCuProperty* udfCuProp;
    //      for(int i = 0; i < deviceNm; ++i){
    //          udfCuGroupName[i] = new char [XRM_MAX_NAME_LEN];
    //          udfCuGroupProp[i] = (xrmUdfCuGroupProperty*)malloc(sizeof(xrmUdfCuGroupProperty));
    //          memset(udfCuGroupProp[i], 0, sizeof(xrmUdfCuGroupProperty));
    //          strcpy(udfCuGroupName[i], (baseName+std::to_string(i)).c_str());
    //          udfCuGroupProp[i]->optionUdfCuListNum = 1;
    //          udfCuListProp = &udfCuGroupProp[i]->optionUdfCuListProps[0];
    //          udfCuListProp->cuNum = cuNm;
    //          udfCuListProp->sameDevice = true;
    //          for (int32_t cuIdx = 0; cuIdx < udfCuListProp->cuNum; cuIdx++) {
    //              std::string cuName0 = kernelName + ":" + kernelName + "_" + std::to_string(i*udfCuListProp->cuNum +
    //              cuIdx);
    //              //std::string cuName0 = kernelName + ":" + kernelName + "_" + std::to_string(cuIdx);
    //              udfCuProp = &udfCuListProp->udfCuProps[cuIdx];
    //              strcpy(udfCuProp->cuName, cuName0.c_str());
    //              udfCuProp->devExcl = false;
    //              udfCuProp->requestLoad = requestLoad;
    //          }
    //          int ret = xrmUdfCuGroupDeclare(ctx, udfCuGroupProp[i], udfCuGroupName[i]);
    //          if (ret == XRM_SUCCESS){
    //              printf("INFO: User defined cu group from same device undeclaration success\n");
    //          }else{
    //              printf("ERROR: User defined cu group from same device undeclaration fail\n");
    //              exit(1);
    //          }
    //          free(udfCuGroupProp[i]);
    //      }
    //  }
    void freeCuGroup(unsigned int deviceNm) {
        int ret = 0;
        for (int i = 0; i < 1; ++i) {
            ret = xrmUdfCuGroupUndeclare(ctx, udfCuGroupName[i]);
            if (ret == XRM_SUCCESS) {
                printf("INFO: User defined cu group from same device undeclaration success\n");
            } else {
                printf("ERROR: User defined cu group from same device undeclaration fail\n");
                exit(1);
            }
            delete[] udfCuGroupName[i];
        }
        delete[] udfCuGroupName;
    }
    void setUpCuGroup(unsigned int deviceNm,
                      unsigned int cuNm,
                      std::string kernelName,
                      std::string kernelAlias,
                      unsigned int requestLoad) {
        udfCuGroupName = new char*[1];
        xrmUdfCuGroupProperty* udfCuGroupProp[1];
        std::string baseName = "udfCuGroupSameDevice";
        xrmUdfCuListProperty* udfCuListProp;
        xrmUdfCuProperty* udfCuProp;
        udfCuGroupName[0] = new char[XRM_MAX_NAME_LEN];
        udfCuGroupProp[0] = (xrmUdfCuGroupProperty*)malloc(sizeof(xrmUdfCuGroupProperty));
        memset(udfCuGroupProp[0], 0, sizeof(xrmUdfCuGroupProperty));
        strcpy(udfCuGroupName[0], baseName.c_str());
        udfCuGroupProp[0]->optionUdfCuListNum = 1;
        udfCuListProp = &udfCuGroupProp[0]->optionUdfCuListProps[0];
        udfCuListProp->cuNum = cuNm * deviceNm;
        udfCuListProp->sameDevice = false;
        for (int32_t cuIdx = 0; cuIdx < udfCuListProp->cuNum; cuIdx++) {
            std::string cuName0 = kernelName + ":" + kernelName + "_" + std::to_string(cuIdx);
            // std::string cuName0 = kernelName + ":" + kernelName + "_" + std::to_string(cuIdx);
            udfCuProp = &udfCuListProp->udfCuProps[cuIdx];
            strcpy(udfCuProp->cuName, cuName0.c_str());
            udfCuProp->devExcl = false;
            udfCuProp->requestLoad = requestLoad;
        }
        int ret = xrmUdfCuGroupDeclare(ctx, udfCuGroupProp[0], udfCuGroupName[0]);
        if (ret == XRM_SUCCESS) {
            printf("INFO: User defined cu group from same device undeclaration success\n");
        } else {
            printf("ERROR: User defined cu group from same device undeclaration fail\n");
            exit(1);
        }
        free(udfCuGroupProp[0]);
    }

    // void unloadXclbin(unsigned int deviceId) { int ret = xrmUnloadOneDevice(ctx, deviceId); }
    void unloadXclbin(unsigned int deviceId) { xrmUnloadOneDevice(ctx, deviceId); }
    void loadXclbin(unsigned int deviceId, char* xclbinName) {
        unloadXclbin(deviceId);
        // int res = xrmLoadOneDevice(ctx, deviceId, xclbinName);
        xrmLoadOneDevice(ctx, deviceId, xclbinName);
    }
    std::thread unloadXclbinNonBlock(unsigned int deviceId) { return std::thread(xrmUnloadOneDevice, ctx, deviceId); }
    std::thread loadXclbinNonBlock(unsigned int deviceId, char* xclbinName) {
        return std::thread(xrmLoadOneDevice, ctx, deviceId, xclbinName);
    }
    std::future<int> loadXclbinAsync(unsigned int deviceId, char* xclbinName) {
        std::future<int> ret = std::async(&xrmLoadOneDevice, ctx, deviceId, xclbinName);
        return ret;
    }

    void allocCU(xrmCuResource* resR, const char* kernelName, const char* kernelAlias, int requestLoad) {
        xrmCuProperty propR;
        memset(&propR, 0, sizeof(xrmCuProperty));
        strcpy(propR.kernelName, kernelName);
        strcpy(propR.kernelAlias, kernelAlias);
        propR.devExcl = false;
        propR.requestLoad = requestLoad;
        propR.poolId = 0;
        uint64_t interval = 1;

        uint32_t ret = xrmCuBlockingAlloc(ctx, &propR, interval, resR);

        if (ret != 0) {
            printf("Error: Fail to alloc cu (xrmCuBlockingAlloc) \n");
        };
    }

    void allocGroupCU(xrmCuGroupResource* resR, std::string groupName) {
        xrmCuGroupProperty cuGroupProp;
        memset(&cuGroupProp, 0, sizeof(xrmCuGroupProperty));
        strcpy(cuGroupProp.udfCuGroupName, groupName.c_str());
        cuGroupProp.poolId = 0;

        // uint64_t interval = 0; // to use the XRM default interval
        uint64_t interval = 4000; // to use the XRM default interval

        int ret = xrmCuGroupBlockingAlloc(ctx, &cuGroupProp, interval, resR);
    }

    xrmCuListResource allocMultiCU(
        const char* kernelName, const char* kernelAlias, int requestLoad, int deviceNumber, int cuNumber) {
        xrmCuListProperty CuListProp;
        xrmCuListResource CuListRes;

        memset(&CuListProp, 0, sizeof(xrmCuListProperty));
        memset(&CuListRes, 0, sizeof(xrmCuListResource));

        std::cout << "request cu number = " << cuNumber << std::endl;
        CuListProp.cuNum = cuNumber * deviceNumber;
        for (int i = 0; i < CuListProp.cuNum; ++i) {
            strcpy(CuListProp.cuProps[i].kernelName, kernelName);
            strcpy(CuListProp.cuProps[i].kernelAlias, kernelAlias);
            CuListProp.cuProps[i].devExcl = false;
            CuListProp.cuProps[i].requestLoad = requestLoad;
            CuListProp.cuProps[i].poolId = 0;
        }
        uint64_t interval = 1;
        uint32_t ret = xrmCuListBlockingAlloc(ctx, &CuListProp, interval, &CuListRes);
        if (ret != 0) {
            printf("Error: Fail to alloc cu (xrmCuListBlockingAlloc) \n");
            memset(&CuListRes, 0, sizeof(xrmCuListResource));
        };

        return CuListRes;
    }

    unsigned int fetchCuInfo(const char* kernelName,
                             const char* kernelAlias,
                             int requestLoad,
                             unsigned int& deviceNm,
                             uint64_t& maxChannelSize,
                             unsigned int& maxCU,
                             unsigned int** deviceID,
                             unsigned int** cuID) {
        xrmCuProperty propR;
        memset(&propR, 0, sizeof(xrmCuProperty));
        strcpy(propR.kernelName, kernelName);
        strcpy(propR.kernelAlias, kernelAlias);
        propR.devExcl = false;
        propR.requestLoad = requestLoad;
        propR.poolId = 0;
        maxCU = xrmCheckCuAvailableNum(ctx, &propR);
        xrmCuListProperty cuListPropR;
        xrmCuListResource cuListResR;
        cuListPropR.cuNum = maxCU;
        unsigned int* devices = new unsigned int[maxCU];
        unsigned int* cus = new unsigned int[maxCU];

        for (int i = 0; i < cuListPropR.cuNum; i++) {
            strcpy(cuListPropR.cuProps[i].kernelName, kernelName);
            strcpy(cuListPropR.cuProps[i].kernelAlias, kernelAlias);
            cuListPropR.cuProps[i].devExcl = false;
            cuListPropR.cuProps[i].requestLoad = requestLoad;
            cuListPropR.cuProps[i].poolId = 0;
        }
        uint32_t alloc0 = xrmCuListAlloc(ctx, &cuListPropR, &cuListResR);

        deviceNm = 0;
        if (alloc0 != 0) {
            printf("Error: Fail to alloc cu list (xrmCuListAlloc) \n");
        } else {
            memBankSizeTransfer(cuListResR.cuResources[0].membankSize, maxChannelSize);
            for (int i = 0; i < cuListResR.cuNum; i++) {
                printf("INFO: Allocated cu list: cu %d\n", i);
                printf("   xclbinFileName is:  %s\n", cuListResR.cuResources[i].xclbinFileName);
                printf("   kernelPluginFileName is:  %s\n", cuListResR.cuResources[i].kernelPluginFileName);
                printf("   kernelName is:  %s\n", cuListResR.cuResources[i].kernelName);
                printf("   kernelAlias is:  %s\n", cuListResR.cuResources[i].kernelAlias);
                printf("   instanceName is:  %s\n", cuListResR.cuResources[i].instanceName);
                printf("   cuName is:  %s\n", cuListResR.cuResources[i].cuName);
                printf("   deviceId is:  %d\n", cuListResR.cuResources[i].deviceId);
                printf("   cuId is:  %d\n", cuListResR.cuResources[i].cuId);
                printf("   channelId is:  %d\n", cuListResR.cuResources[i].channelId);
                printf("   cuType is:  %d\n", cuListResR.cuResources[i].cuType);
                printf("   baseAddr is:  0x%lx\n", cuListResR.cuResources[i].baseAddr);
                printf("   membankId is:  %d\n", cuListResR.cuResources[i].membankId);
                printf("   membankType is:  %d\n", cuListResR.cuResources[i].membankType);
                printf("   membankSize is:  %ld Byte\n", maxChannelSize);
                printf("   membankBaseAddr is:  0x%lx\n", cuListResR.cuResources[i].membankBaseAddr);
                printf("   allocServiceId is:  %lu\n", cuListResR.cuResources[i].allocServiceId);
                printf("   poolId is:  %lu\n", cuListResR.cuResources[i].poolId);
                printf("   channelLoad is:  %d\n", cuListResR.cuResources[i].channelLoad);
                cus[i] = cuListResR.cuResources[i].cuId;
                devices[i] = cuListResR.cuResources[i].deviceId;
                bool flag = false;
                for (int j = 0; j < i; ++j) {
                    if (devices[j] == (unsigned int)cuListResR.cuResources[i].deviceId) {
                        flag = true;
                    }
                }
                if (flag == false) {
                    deviceNm += 1;
                }
            }
            *cuID = cus;
            *deviceID = devices;
            assert(deviceNm != 0);
            if (xrmCuListRelease(ctx, &cuListResR))
                printf("INFO: Success to release cu list\n");
            else
                printf("Error: Fail to release cu list\n");
        }
        std::cout << "INFO: Available device number = " << deviceNm << std::endl;
        std::cout << "INFO: Available CU number = " << maxCU << std::endl;
    }

    void freeXRM() {
        if (xrmDestroyContext(ctx) != XRM_SUCCESS)
            printf("INFO: Destroy context failed\n");
        else
            printf("INFO: Destroy context success\n");
    };

    xrmContext* ctx;

   private:
    void memBankSizeTransfer(uint64_t input, uint64_t& output) {
        std::stringstream str;
        std::string pre = std::bitset<64>(input).to_string();
        str << pre;
        str >> std::hex >> output;
    };
};

/**
 * Type erased std::packaged_task<RT()>
 *
 * Wraps a std::packaged_task of any return type, such that the task's
 * return value can be captured in a future.
 *
 * Objects of this task class can be stored in any STL container even
 * when the underlying std::packaged_tasks are of different types.
 */
class task {
    struct task_iholder {
        virtual ~task_iholder(){};
        virtual void execute(unsigned int ID,
                             unsigned int ID2,
                             unsigned int ID3,
                             class openXRM* xrm,
                             xrmCuResource* resR,
                             std::string instance) = 0;
    };

    template <typename Callable>
    struct task_holder : public task_iholder {
        Callable held;
        task_holder(Callable&& t) : held(std::move(t)) {}
        void execute(unsigned int ID,
                     unsigned int ID2,
                     unsigned int ID3,
                     class openXRM* xrm,
                     xrmCuResource* resR,
                     std::string instance) {
            std::thread w1(std::move(held), ID, ID2, ID3, xrm->ctx, resR, instance);
            w1.detach();
        }
    };

    std::unique_ptr<task_iholder> content;

   public:
    task() : content(nullptr) {}

    task(task&& rhs) : content(std::move(rhs.content)) {}

    template <typename Callable>
    task(Callable&& c) : content(new task_holder<Callable>(std::forward<Callable>(c))) {}

    task& operator=(task&& rhs) {
        content = std::move(rhs.content);
        return *this;
    }

    bool valid() const { return content != nullptr; }

    void execute(unsigned int ID,
                 unsigned int ID2,
                 unsigned int ID3,
                 class openXRM* xrm,
                 xrmCuResource* resR,
                 std::string instance) {
        content->execute(ID, ID2, ID3, xrm, resR, instance);
    }
};

/**
 * Multiple producer / multiple consumer queue of task objects
 *
 * This code is not specifically tied to task::task, but we keep
 * the defintion here to make task.h stand-alone
 */
template <typename Task>
class mpmcqueue {
    std::queue<Task> m_tasks;
    mutable std::mutex m_mutex;
    std::condition_variable m_work;
    bool m_stop = false;
    unsigned long tp = 0;       // time point when last task consumed
    unsigned long waittime = 0; // wait time from tp to next task avail
    bool debug = false;

   public:
    mpmcqueue() {}

    explicit mpmcqueue(bool dbg) : debug(dbg) {}

    void addWork(Task&& t) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_tasks.push(std::move(t));
        m_work.notify_one();
    }
    bool empty() { return m_tasks.empty(); }

    Task getWork() {
        std::unique_lock<std::mutex> lk(m_mutex);
        while (!m_stop && m_tasks.empty()) {
            m_work.wait(lk);
        }

        Task task;
        if (!m_stop) {
            task = std::move(m_tasks.front());
            m_tasks.pop();
        }
        return task;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_tasks.size();
    }

    void stop() {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_stop = true;
        m_work.notify_all();
    }
};

using queue = mpmcqueue<task>;

/**
 * event class wraps std::future<RT>
 *
 * Adds a ready() function that can be used to poll if event is ready.
 * Otherwise, currently adds no value compared to bare std::future
 */
template <typename RT>
class event {
   public:
    typedef RT value_type;
    typedef std::future<value_type> FutureType;

   private:
    mutable FutureType m_future;

   public:
    event() = delete;
    event(const event& rhs) = delete;

    event(const event&& rhs) : m_future(std::move(rhs.m_future)) {}

    event(FutureType&& f) : m_future(std::forward<FutureType>(f)) {}

    event& operator=(event&& rhs) {
        m_future = std::move(rhs.m_future);
        return *this;
    }

    RT wait() const { return m_future.get(); }

    RT get() const { return m_future.get(); }

    bool ready() const { return (m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready); }
};

/**
 * Functions for adding work (tasks) to a task queue.
 *
 * All functions return a task::event that encapsulates the
 * return type of the task.
 *
 * Variants of the functions supports adding both free functions
 * and member functions associated with some class object.
 */
// Free function, lambda, functor

template <typename Q, typename F, typename... Args>
auto createL3(Q& q, F&& f, Args&&... args) -> event<int> {
    typedef std::packaged_task<int(int, int, int, xrmContext*, xrmCuResource*, std::string)> task_type;
    task_type t(std::bind(std::forward<F>(f), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                          std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                          std::forward<Args>(args)...));
    event<int> e(t.get_future());
    q.addWork(std::move(t));
    return e;
}

inline void worker(queue& q,
                   class openXRM* xrm,
                   std::string kernelName,
                   std::string kernelAlias,
                   unsigned int requestLoad,
                   unsigned int deviceNm,
                   unsigned int cuNm) {
    int requestNm = deviceNm * cuNm;
    xrmCuResource* resR[requestNm];
#ifdef __DEBUG__
    int requestCnt = 0;
#endif
    while (true) {
#ifdef __DEBUG__
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_start_compute =
            std::chrono::high_resolution_clock::now();
#endif
        class task t[requestNm];
        for (int i = 0; i < requestNm; ++i) {
            t[i] = q.getWork();
            resR[i] = (xrmCuResource*)malloc(sizeof(xrmCuResource));
            memset(resR[i], 0, sizeof(xrmCuResource));
        }

        bool toStop = false;
        for (int i = 0; i < requestNm; ++i) {
            if (!t[i].valid()) toStop = true;
        }
        if (toStop) break;
#ifdef __DEBUG__
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_compute_time =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> l_durationSec = l_tp_compute_time - l_tp_start_compute;
        double l_timeMs = l_durationSec.count() * 1e3;
        std::cout << "INFO: getwork time =  " << std::fixed << std::setprecision(6) << l_timeMs << " msec" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
#endif

#ifdef __DEBUG__
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_start_compute2 =
            std::chrono::high_resolution_clock::now();
#endif
        for (int i = 0; i < requestNm; ++i) {
            xrm->allocCU(resR[i], kernelName.c_str(), kernelAlias.c_str(), requestLoad);
        }

#ifdef __DEBUG__
        std::cout << "INFO: Allocated deviceID = " << deviceID << "\t cuID = " << cuID << "\t channelID = " << channelID
                  << "\t instance name = " << instanceName << "\t request ID = " << requestCnt << "\t number per while "
                  << requestNm << std::endl;
#endif

        for (int i = 0; i < requestNm; i++) {
            unsigned int deviceID = i / cuNm;
            unsigned int cuID = i % cuNm;
            unsigned int ID;
            for (int j = 0; j < requestNm; ++j) {
                if ((deviceID == resR[j][0].deviceId) && (cuID == resR[j][0].cuId)) {
                    ID = j;
                }
            }
            unsigned int channelID = resR[ID][0].channelId;
            std::string instanceName = resR[ID][0].instanceName;
            t[i].execute(deviceID, cuID, channelID, xrm, resR[ID], instanceName);
        }
#ifdef __DEBUG__
        requestCnt++;
#endif

#ifdef __DEBUG__
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_compute_time2 =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> l_durationSec2 = l_tp_compute_time2 - l_tp_start_compute2;
        double l_timeMs2 = l_durationSec2.count() * 1e3;
        std::cout << "INFO: Cu allocation time =  " << std::fixed << std::setprecision(6) << l_timeMs2 << " msec"
                  << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
#endif
    }
}

inline void worker2(queue& q,
                    class openXRM* xrm,
                    std::string kernelName,
                    std::string kernelAlias,
                    unsigned int requestLoad,
                    unsigned int deviceNm,
                    unsigned int cuNm) {
    int requestNm = deviceNm * cuNm;
    xrm->setUpCuGroup(deviceNm, cuNm, kernelName, kernelAlias, requestLoad);

    int requestCnt = 0;
    int iteration = 100;
    xrmCuGroupResource* resR[iteration][1];
    for (int i = 0; i < iteration; ++i) {
        for (int j = 0; j < 1; ++j) {
            resR[i][j] = (xrmCuGroupResource*)malloc(sizeof(xrmCuGroupResource));
            memset(resR[i][j], 0, sizeof(xrmCuGroupResource));
        }
    }
    xrmCuResource* cuRes[requestNm];
    while (true) {
#ifdef __DEBUG__
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_start_compute =
            std::chrono::high_resolution_clock::now();
#endif
        class task t[requestNm];
        for (int i = 0; i < requestNm; ++i) {
            t[i] = q.getWork();
        }
        bool toStop = false;
        for (int i = 0; i < requestNm; ++i) {
            if (!t[i].valid()) toStop = true;
        }
        if (toStop) break;
#ifdef __DEBUG__
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_compute_time =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> l_durationSec = l_tp_compute_time - l_tp_start_compute;
        double l_timeMs = l_durationSec.count() * 1e3;
        std::cout << "INFO: getwork time =  " << std::fixed << std::setprecision(6) << l_timeMs << " msec" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
#endif

#ifdef __DEBUG__
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_start_compute2 =
            std::chrono::high_resolution_clock::now();
#endif
        for (int i = 0; i < 1; ++i) {
            xrm->allocGroupCU(resR[requestCnt % iteration][i], xrm->udfCuGroupName[0]);
        }

        for (int i = 0; i < 1; ++i) {
            for (int j = 0; j < cuNm * deviceNm; ++j) {
                cuRes[j + i * cuNm] = &(resR[requestCnt % iteration][i]->cuResources[j]);
                unsigned int deviceID = cuRes[j + i * cuNm]->deviceId;
                unsigned int cuID = cuRes[j + i * cuNm]->cuId;
                unsigned int channelID = cuRes[j + i * cuNm]->channelId;
                std::string instanceName = cuRes[j + i * cuNm]->instanceName;
                t[i * cuNm + j].execute(deviceID, cuID, channelID, xrm, cuRes[j + i * cuNm], instanceName);
            }
        }
        requestCnt++;
#ifdef __DEBUG__
        std::cout << "INFO: Allocated deviceID = " << deviceID << "\t cuID = " << cuID << "\t channelID = " << channelID
                  << "\t instance name = " << instanceName << "\t request ID = " << requestCnt << "\t number per while "
                  << requestNm << std::endl;
        std::chrono::time_point<std::chrono::high_resolution_clock> l_tp_compute_time2 =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> l_durationSec2 = l_tp_compute_time2 - l_tp_start_compute2;
        double l_timeMs2 = l_durationSec2.count() * 1e3;
        std::cout << "INFO: Cu allocation time =  " << std::fixed << std::setprecision(6) << l_timeMs2 << " msec"
                  << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
#endif
    }
}
} // L3
} // graph
} // xf

#endif

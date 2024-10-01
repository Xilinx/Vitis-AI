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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#ifndef wmb
#ifndef ZCU102
#define wmb()
#else
#define wmb() asm volatile("dmb ishst" ::: "memory")
#endif
#endif

using namespace std;

template <typename T, typename... Args>
std::shared_ptr<T> Singleton(Args &&... args) {
  static std::weak_ptr<T> singleton;
  auto ret = singleton.lock();
  if (ret) {
    return ret;
  }
  ret = std::make_shared<T>(std::forward<Args>(args)...);
  singleton = ret;
  return ret;
}
struct DevMem {
  DevMem() {
    dev_fd = open("/dev/mem", O_RDWR | O_SYNC);
    assert(dev_fd > 0);
  }
  ~DevMem() {
    assert(dev_fd > 0);
    close(dev_fd);
  }
  int dev_fd;
};
struct PhysicalMemory {
  PhysicalMemory(uintptr_t phy_base_v, size_t length_v)
      : dev_mem(Singleton<DevMem>()), length(length_v), phy_base(phy_base_v) {
    map_base = (mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED,
                     dev_mem->dev_fd, phy_base));
    if (0)
      cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "] "
           << "map_base " << map_base << " "
           << "phy_base " << hex << "0x" << phy_base << dec << " " << endl;
    assert(map_base != MAP_FAILED);
  }
  ~PhysicalMemory() {
    if (map_base != MAP_FAILED) {
      int r = munmap(map_base, length);
      if (!(r == 0)) {
        assert(false);
      }
    }
  }
  template <class Tp>
  Tp *data(size_t offset = 0) {
    return (Tp *)((uintptr_t)(map_base) + offset);
  }
  template <class Tp>
  Tp *PhyAddr(uintptr_t addr) {
    if (!(addr >= phy_base) || !(addr < phy_base + length)) {
      cerr << "out of range " << hex << "addr = "
           << "0x" << addr << " "
           << "phy_base = "
           << "0x" << phy_base << " "
           << "lenght = "
           << "0x" << length << " " << endl;
      assert(false);
    }
    char *ret = ((char *)map_base) + addr - phy_base;
    return (Tp *)ret;
  }

  void Dump(std::string filename, uintptr_t v_addr, size_t size) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp == NULL) {
      perror("cannot open file:");
      std::cerr << "file = " << filename << endl;
      exit(1);
    }
    Dump(fp, v_addr, size);
    fclose(fp);
  }
  void Dump(FILE *fp, uintptr_t v_addr, size_t size) {
    char buf[sizeof(uint32_t)];
    uint32_t *p32 = (uint32_t *)buf;
    uintptr_t addr = v_addr;
    size_t addr_aligned = addr / sizeof(uint32_t) * sizeof(uint32_t);
    uint32_t *from4 = PhyAddr<uint32_t>(addr_aligned);
    unsigned int c = 0;
    while (c < size) {
      *p32 = *from4++;
      for (unsigned int i = addr % sizeof(uint32_t);
           i < sizeof(uint32_t) && c < size; ++i, ++c) {
        fputc(buf[i], fp);
        addr++;
      }
    }
  }
  void DumpToMemory(void *p, uintptr_t v_addr, size_t size) {
    char buf[sizeof(uint32_t)];
    char *dst = (char *)p;
    uint32_t *p32 = (uint32_t *)buf;
    uintptr_t addr = v_addr;
    size_t addr_aligned = addr / sizeof(uint32_t) * sizeof(uint32_t);
    uint32_t *from4 = PhyAddr<uint32_t>(addr_aligned);
    unsigned int c = 0;
    while (c < size) {
      *p32 = *from4++;
      for (unsigned int i = addr % sizeof(uint32_t);
           i < sizeof(uint32_t) && c < size; ++i, ++c) {
        *dst++ = buf[i];
        addr++;
      }
    }
  }
  void CopyFrom(const void *vp, uintptr_t v_addr, size_t size) {
    const char *src = (const char *)vp;
    char buf[sizeof(uint32_t)];
    uint32_t *p32 = (uint32_t *)buf;
    uintptr_t addr = v_addr;
    size_t addr_aligned = addr / sizeof(uint32_t) * sizeof(uint32_t);
    uint32_t *from4 = PhyAddr<uint32_t>(addr_aligned);
    unsigned int c = 0;
    while (c < size) {
      auto start = addr % sizeof(uint32_t);
      *p32 = *from4;
      for (unsigned int i = start; i < sizeof(uint32_t) && c < size; ++i, ++c) {
        buf[i] = *src++;
        addr++;
      }
      *from4++ = *p32;
    }
  }
  void Load(FILE *fp, uintptr_t v_addr, size_t size) {
    char buf[sizeof(uint32_t)];
    uint32_t *p32 = (uint32_t *)buf;
    uintptr_t addr = v_addr;
    size_t addr_aligned = addr / sizeof(uint32_t) * sizeof(uint32_t);
    uint32_t *from4 = PhyAddr<uint32_t>(addr_aligned);
    unsigned int c = 0;
    while (c < size) {
      auto start = addr % sizeof(uint32_t);
      *p32 = *from4;
      if (0)
        std::cout << "buf[0] " << hex << "0x" << (int)buf[0] << dec << " "
                  << "buf[1] " << hex << "0x" << (int)buf[1] << dec << " "
                  << "buf[2] " << hex << "0x" << (int)buf[2] << dec << " "
                  << "buf[3] " << hex << "0x" << (int)buf[3] << dec << " "
                  << "from4 " << (void *)from4 << " " << endl;
      for (unsigned int i = start; i < sizeof(uint32_t) && c < size; ++i, ++c) {
        int ch = fgetc(fp);
        if (ch == EOF) {
          std::cerr << "unexpected file lenght. " << hex << "phy addr = 0x"
                    << addr << " "
                    << "size = 0x " << size << " "
                    << "c = 0x" << c << std::endl;
          exit(1);
        }
        if (0)
          std::cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                    << "] "
                    << "i " << i << " "
                    << "c " << c << " "
                    << "ch " << hex << "0x" << ch << dec << " "
                    << "buf[i] " << hex << "0x" << (int)buf[i] << dec << " "
                    << endl;
        buf[i] = (char)ch;
        addr++;
      }
      if (0)
        std::cout << "buf[0] " << hex << "0x" << (int)buf[0] << dec << " "
                  << "buf[1] " << hex << "0x" << (int)buf[1] << dec << " "
                  << "buf[2] " << hex << "0x" << (int)buf[2] << dec << " "
                  << "buf[3] " << hex << "0x" << (int)buf[3] << dec << " "
                  << endl;
      *from4++ = *p32;
    }
  }
  size_t Verify(FILE *fp, uintptr_t v_addr, size_t size) {
    char buf[sizeof(uint32_t)];
    uint32_t *p32 = (uint32_t *)buf;
    uintptr_t addr = v_addr;
    size_t addr_aligned = addr / sizeof(uint32_t) * sizeof(uint32_t);
    uint32_t *from4 = PhyAddr<uint32_t>(addr_aligned);
    size_t c = 0;
    size_t diff = 0;
    while (c < size) {
      *p32 = *from4++;
      for (unsigned int i = addr % sizeof(uint32_t);
           i < sizeof(uint32_t) && c < size; ++i, ++c) {
        int ch = fgetc(fp);
        if (ch == EOF) {
          std::cerr << "unexpected file lenght. " << hex << "phy addr = 0x"
                    << addr << " "
                    << "size = 0x " << size << " "
                    << "c = 0x" << c << std::endl;
          exit(1);
        }
        if (buf[i] != (char)ch) {
          diff++;
        }
        addr++;
      }
    }
    if (diff != 0) {
      std::cout << "total " << diff << "/" << size << " differences."
                << std::endl;
    }
    return diff;
  }
  std::shared_ptr<DevMem> dev_mem;
  size_t length;
  uintptr_t phy_base;
  void *map_base;
};

template <class _Tp>
struct MemRegValue {
  MemRegValue(PhysicalMemory *phy_mem, off_t offset)
      : pointer_(phy_mem->data<_Tp>(offset)) {}
  operator _Tp() { return read(); }
  _Tp operator=(_Tp v) {
    std::lock_guard<std::mutex> lock(mtx);
    wmb();
    _Tp ret = (*pointer_ = v);
    wmb();
    if (0)
      cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "] "
           << "write ret = " << hex << "0x" << ret << dec << "@"
           << (void *)pointer_ << endl;
    return ret;
  }
  _Tp read() {
    std::lock_guard<std::mutex> lock(mtx);
    wmb();
    _Tp ret = *pointer_;
    wmb();
    if (0)
      cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "] "
           << "read ret = " << hex << "0x" << ret << dec << "@"
           << (void *)pointer_ << endl;
    return ret;
  }

 private:
  using pointer_type = volatile _Tp *;
  pointer_type pointer_;
  static std::mutex mtx;
};
template <class _Tp>
struct std::mutex MemRegValue<_Tp>::mtx;
// static void LoadDataAsTxt(const char *filename, volatile void *data,
//                           int count) {
//     FILE *fp = NULL;
//     fp = fopen(filename, "rt");
//     assert(fp != NULL);
//     std::cout << "load input to " << data << " ";
//     cout << "input: ";
//     for (int i = 0; i < count; ++i) {
//         int c;
//         int r = fscanf(fp, "%d\n", &c);
//         if (r != 1) {
//             cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ <<
//             "] "
//                  << "cannot read "
//                  << " " << filename << " " << endl;
//             exit(1);
//         }
//         ((volatile signed char *)data)[i] = (signed char)c;
//         if (i < 10) {
//             cout << " " << c;
//             cout.flush();
//         }
//     }
//     fclose(fp);
//     cout << endl;
// }
// static void DumpDataAsTxt(const char *filename, void *data, int count) {
//     FILE *fp = fopen(filename, "wt");
//     assert(fp != NULL);
//     std::cout << "output: ";
//     for (int i = 0; i < count; ++i) {
//         int c = (int)(((signed char *)data)[i]);
//         if (i < 10) {
//             std::cout << " " << c;
//         }
//         fprintf(fp, "%d\n", c);
//     }
//     std::cout << std::endl;
//     fclose(fp);
// }

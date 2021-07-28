/*
 * Copyright (C) 2015-2019, Xilinx Inc - All rights reserved
 * Xilinx Runtime (XRT) APIs
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef _XCL_XRT_CORE_H_
#define _XCL_XRT_CORE_H_

#ifdef __cplusplus
#include <cstdlib>
#include <cstdint>
#else
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#if defined(_WIN32)
#ifdef XCL_DRIVER_DLL_EXPORT
#define XCL_DRIVER_DLLESPEC __declspec(dllexport)
#else
#define XCL_DRIVER_DLLESPEC __declspec(dllimport)
#endif
#else
#define XCL_DRIVER_DLLESPEC __attribute__((visibility("default")))
#endif


#include "xclbin.h"
#include "xclperf.h"
#include "xcl_app_debug.h"
#include "xclerr.h"
#include "xclhal2_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DOC: Xilinx Runtime (XRT) Library Interface Definitions
 *
 * Header file *xrt.h* defines data structures and function signatures exported by
 * Xilinx Runtime (XRT) Library. XRT is part of software stack which is integrated
 * into Xilinx reference platform.
 */

/**
 * typedef xclDeviceHandle - opaque device handle
 *
 * A device handle of xclDeviceHandle kind is obtained by opening a device. Clients pass this
 * device handle to refer to the opened device in all future interaction with XRT.
 */
typedef void * xclDeviceHandle;

struct axlf;

/**
 * Structure used to obtain various bits of information from the device.
 */

struct xclDeviceInfo2 {
  unsigned mMagic; // = 0X586C0C6C; XL OpenCL X->58(ASCII), L->6C(ASCII), O->0 C->C L->6C(ASCII);
  char mName[256];
  unsigned short mHALMajorVersion;
  unsigned short mHALMinorVersion;
  unsigned short mVendorId;
  unsigned short mDeviceId;
  unsigned short mSubsystemId;
  unsigned short mSubsystemVendorId;
  unsigned short mDeviceVersion;
  size_t mDDRSize;                    // Size of DDR memory
  size_t mDataAlignment;              // Minimum data alignment requirement for host buffers
  size_t mDDRFreeSize;                // Total unused/available DDR memory
  size_t mMinTransferSize;            // Minimum DMA buffer size
  unsigned short mDDRBankCount;
  unsigned short mOCLFrequency[4];
  unsigned short mPCIeLinkWidth;
  unsigned short mPCIeLinkSpeed;
  unsigned short mDMAThreads;
  unsigned short mOnChipTemp;
  unsigned short mFanTemp;
  unsigned short mVInt;
  unsigned short mVAux;
  unsigned short mVBram;
  float mCurrent;
  unsigned short mNumClocks;
  unsigned short mFanSpeed;
  bool mMigCalib;
  unsigned long long mXMCVersion;
  unsigned long long mMBVersion;
  unsigned short m12VPex;
  unsigned short m12VAux;
  unsigned long long mPexCurr;
  unsigned long long mAuxCurr;
  unsigned short mFanRpm;
  unsigned short mDimmTemp[4];
  unsigned short mSE98Temp[4];
  unsigned short m3v3Pex;
  unsigned short m3v3Aux;
  unsigned short mDDRVppBottom;
  unsigned short mDDRVppTop;
  unsigned short mSys5v5;
  unsigned short m1v2Top;
  unsigned short m1v8Top;
  unsigned short m0v85;
  unsigned short mMgt0v9;
  unsigned short m12vSW;
  unsigned short mMgtVtt;
  unsigned short m1v2Bottom;
  unsigned long long mDriverVersion;
  unsigned mPciSlot;
  bool mIsXPR;
  unsigned long long mTimeStamp;
  char mFpga[256];
  unsigned short mPCIeLinkWidthMax;
  unsigned short mPCIeLinkSpeedMax;
  unsigned short mVccIntVol;
  unsigned short mVccIntCurr;
  unsigned short mNumCDMA;
};

/**
 *  Unused, keep for backwards compatibility
 */
enum xclBOKind {
    XCL_BO_SHARED_VIRTUAL = 0,
    XCL_BO_SHARED_PHYSICAL,
    XCL_BO_MIRRORED_VIRTUAL,
    XCL_BO_DEVICE_RAM,
    XCL_BO_DEVICE_BRAM,
    XCL_BO_DEVICE_PREALLOCATED_BRAM,
};

enum xclBOSyncDirection {
    XCL_BO_SYNC_BO_TO_DEVICE = 0,
    XCL_BO_SYNC_BO_FROM_DEVICE,
};

/**
 * Define address spaces on the device AXI bus. The enums are used in xclRead() and xclWrite()
 * to pass relative offsets.
 */

enum xclAddressSpace {
    XCL_ADDR_SPACE_DEVICE_FLAT = 0,     // Absolute address space
    XCL_ADDR_SPACE_DEVICE_RAM = 1,      // Address space for the DDR memory
    XCL_ADDR_KERNEL_CTRL = 2,           // Address space for the OCL Region control port
    XCL_ADDR_SPACE_DEVICE_PERFMON = 3,  // Address space for the Performance monitors
    XCL_ADDR_SPACE_DEVICE_REG     = 4,  // Address space for device registers.
    XCL_ADDR_SPACE_DEVICE_CHECKER = 5,  // Address space for protocol checker
    XCL_ADDR_SPACE_MAX = 8
};


/**
 * Defines log message severity levels for messages sent to log file with xclLogMsg cmd
 */

enum xrtLogMsgLevel {
     XRT_EMERGENCY = 0,
     XRT_ALERT = 1,
     XRT_CRITICAL = 2,
     XRT_ERROR = 3,
     XRT_WARNING = 4,
     XRT_NOTICE = 5,
     XRT_INFO = 6,
     XRT_DEBUG = 7
};

/**
 * Defines verbosity levels which are passed to xclOpen during device creation time
 */

enum xclVerbosityLevel {
    XCL_QUIET = 0,
    XCL_INFO = 1,
    XCL_WARN = 2,
    XCL_ERROR = 3
};

enum xclResetKind {
    XCL_RESET_KERNEL, // not implemented through xocl user pf
    XCL_RESET_FULL,   // not implemented through xocl user pf
    XCL_USER_RESET
};

struct xclDeviceUsage {
    size_t h2c[8];
    size_t c2h[8];
    size_t ddrMemUsed[8];
    unsigned ddrBOAllocated[8];
    unsigned totalContexts;
    uint64_t xclbinId[4];
    unsigned dma_channel_cnt;
    unsigned mm_channel_cnt;
    uint64_t memSize[8];
};

struct xclBOProperties {
    uint32_t handle;
    uint32_t flags;
    uint64_t size;
    uint64_t paddr;
    int reserved; // not implemented
};

#define	NULLBO	0xffffffff

/**
 * DOC: XRT Device Management APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * xclProbe() - Enumerate devices found in the system
 *
 * Return: count of devices found
 */
XCL_DRIVER_DLLESPEC unsigned xclProbe();

/**
 * xclOpen() - Open a device and obtain its handle.
 *
 * @deviceIndex:   Slot number of device 0 for first device, 1 for the second device...
 * @logFileName:   Log file to use for optional logging
 * @level:         Severity level of messages to log
 *
 * Return:         Device handle
 */
XCL_DRIVER_DLLESPEC xclDeviceHandle xclOpen(unsigned deviceIndex, const char *logFileName,
                                            enum xclVerbosityLevel level);

/**
 * xclClose() - Close an opened device
 *
 * @handle:        Device handle
 */
XCL_DRIVER_DLLESPEC void xclClose(xclDeviceHandle handle);

/**
 * xclResetDevice() - Reset a device or its CL
 *
 * @handle:        Device handle
 * @kind:          Reset kind
*  Return:         0 on success or appropriate error number
 *
 * Reset the device. All running kernels will be killed and buffers in DDR will be
 * purged. A device may be reset if a user's application dies without waiting for
 * running kernel(s) to finish.
 * NOTE: Only implemeted Reset kind through user pf is XCL_USER_RESET
 */
XCL_DRIVER_DLLESPEC int xclResetDevice(xclDeviceHandle handle, enum xclResetKind kind);


/**
 * xclGetDeviceInfo2() - Obtain various bits of information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC int xclGetDeviceInfo2(xclDeviceHandle handle, struct xclDeviceInfo2 *info);

/**
 * xclGetUsageInfo() - Obtain usage information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC int xclGetUsageInfo(xclDeviceHandle handle, struct xclDeviceUsage *info);

/**
 * xclGetErrorStatus() - Obtain error information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC int xclGetErrorStatus(xclDeviceHandle handle, struct xclErrorStatus *info);

/**
 * xclLoadXclBin() - Download FPGA image (xclbin) to the device
 *
 * @handle:        Device handle
 * @buffer:        Pointer to device image (xclbin) in memory
 * Return:         0 on success or appropriate error number
 *
 * Download FPGA image (AXLF) to the device. The PR bitstream is encapsulated inside
 * xclbin as a section. xclbin may also contains other sections which are suitably
 * handled by the driver.
 */
XCL_DRIVER_DLLESPEC int xclLoadXclBin(xclDeviceHandle handle, const struct axlf *buffer);
/**
 * xclGetSectionInfo() - Get Information from sysfs about the downloaded xclbin sections
 *
 * @handle:        Device handle
 * @info:          Pointer to preallocated memory which will store the return value.
 * @size:          Pointer to preallocated memory which will store the return size.
 * @kind:          axlf_section_kind for which info is being queried
 * @index:         The (sub)section index for the "kind" type.
 * Return:         0 on success or appropriate error number
 *
 * Get the section information from sysfs. The index corrresponds to the (section) entry
 * of the axlf_section_kind data being queried. The info and the size contain the return
 * binary value of the subsection and its size.
 */

XCL_DRIVER_DLLESPEC int xclGetSectionInfo(xclDeviceHandle handle, void* info,
	size_t *size, enum axlf_section_kind kind, int index);

/**
 * xclReClock2() - Configure PR region frequncies
 *
 * @handle:        Device handle
 * @region:        PR region (always 0)
 * @targetFreqMHz: Array of target frequencies in order for the Clock Wizards driving
 *                 the PR region
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC int xclReClock2(xclDeviceHandle handle, unsigned short region,
                                    const unsigned short *targetFreqMHz);

/**
 * xclLockDevice() - Get exclusive ownership of the device
 *
 * @handle:        Device handle
 * Return:         0 on success or appropriate error number
 *
 * The lock is necessary before performing buffer migration, register access or
 * bitstream downloads.
 */
XCL_DRIVER_DLLESPEC int xclLockDevice(xclDeviceHandle handle);

/**
 * xclUnlockDevice() - Release exclusive ownership of the device
 *
 * @handle:        Device handle
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC int xclUnlockDevice(xclDeviceHandle handle);

/**
 * xclOpenContext() - Create shared/exclusive context on compute units
 *
 * @handle:        Device handle
 * @xclbinId:      UUID of the xclbin image running on the device
 * @ipIndex:       IP/CU index in the IP LAYOUT array
 * @shared:        Shared access or exclusive access
 * Return:         0 on success or appropriate error number
 *
 * The context is necessary before submitting execution jobs using xclExecBuf(). Contexts may be
 * exclusive or shared. Allocation of exclusive contexts on a compute unit would succeed
 * only if another client has not already setup up a context on that compute unit. Shared
 * contexts can be concurrently allocated by many processes on the same compute units.
 */
XCL_DRIVER_DLLESPEC int xclOpenContext(xclDeviceHandle handle, uuid_t xclbinId, unsigned int ipIndex,
                                       bool shared);

/**
 * xclCloseContext() - Close previously opened context
 *
 * @handle:        Device handle
 * @xclbinId:      UUID of the xclbin image running on the device
 * @ipIndex:       IP/CU index in the IP LAYOUT array
 * Return:         0 on success or appropriate error number
 *
 * Close a previously allocated shared/exclusive context for a compute unit.
 */
XCL_DRIVER_DLLESPEC int xclCloseContext(xclDeviceHandle handle, uuid_t xclbinId, unsigned ipIndex);

/*
 * Update the device BPI PROM with new image
 */
XCL_DRIVER_DLLESPEC int xclUpgradeFirmware(xclDeviceHandle handle, const char *fileName);

/*
 * Update the device PROM with new image with clearing bitstream
 */
XCL_DRIVER_DLLESPEC int xclUpgradeFirmware2(xclDeviceHandle handle, const char *file1, const char* file2);

/*
 * Update the device SPI PROM with new image
 */
XCL_DRIVER_DLLESPEC int xclUpgradeFirmwareXSpi(xclDeviceHandle handle, const char *fileName, int index);

/**
 * xclBootFPGA() - Boot the FPGA from PROM
 *
 * @handle:        Device handle
 * Return:         0 on success or appropriate error number
 *
 * This should only be called when there are no other clients. It will cause PCIe bus re-enumeration
 */
XCL_DRIVER_DLLESPEC int xclBootFPGA(xclDeviceHandle handle);

/*
 * Write to /sys/bus/pci/devices/<deviceHandle>/remove and initiate a pci rescan by
 * writing to /sys/bus/pci/rescan.
 */
XCL_DRIVER_DLLESPEC int xclRemoveAndScanFPGA();

/*
 * Get the version number. 1 => Hal1 ; 2 => Hal2
 */
XCL_DRIVER_DLLESPEC unsigned int xclVersion();

/* End XRT Device Management APIs */


/**
 * xclLogMsg() - Send message to log file as per settings in ini file.
 *
 * @handle:        Device handle
 * @level:         Severity level of the msg
 * @tag:           Tag supplied by the client, like "OCL", "XMA", etc.
 * @format:        Format of Msg string to write to log file
 * @...:           All other arguments as per the format
 *
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC int xclLogMsg(xclDeviceHandle handle, enum xrtLogMsgLevel level, const char* tag, const char* format, ...);

/**
 * DOC: XRT Buffer Management APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Buffer management APIs are used for managing device memory and migrating buffers
 * between host and device memory
 */

/**
 * xclAllocBO() - Allocate a BO of requested size with appropriate flags
 *
 * @handle:        Device handle
 * @size:          Size of buffer
 * @unused:        This argument is ignored
 * @flags:         Specify bank information, etc
 * Return:         BO handle
 */
XCL_DRIVER_DLLESPEC unsigned int xclAllocBO(xclDeviceHandle handle, size_t size,
       	int unused, unsigned flags);

/**
 * xclAllocUserPtrBO() - Allocate a BO using userptr provided by the user
 *
 * @handle:        Device handle
 * @userptr:       Pointer to 4K aligned user memory
 * @size:          Size of buffer
 * @flags:         Specify bank information, etc
 * Return:         BO handle
 */
XCL_DRIVER_DLLESPEC unsigned int xclAllocUserPtrBO(xclDeviceHandle handle,
	void *userptr, size_t size, unsigned flags);

/**
 * xclFreeBO() - Free a previously allocated BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 */
XCL_DRIVER_DLLESPEC void xclFreeBO(xclDeviceHandle handle, unsigned int boHandle);

/**
 * xclWriteBO() - Copy-in user data to host backing storage of BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @src:           Source data pointer
 * @size:          Size of data to copy
 * @seek:          Offset within the BO
 * Return:         0 on success or appropriate error number
 *
 * Copy host buffer contents to previously allocated device memory. ``seek`` specifies how many bytes
 * to skip at the beginning of the BO before copying-in ``size`` bytes of host buffer.
 */
XCL_DRIVER_DLLESPEC size_t xclWriteBO(xclDeviceHandle handle, unsigned int boHandle,
                                       const void *src, size_t size, size_t seek);

/**
 * xclReadBO() - Copy-out user data from host backing storage of BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @dst:           Destination data pointer
 * @size:          Size of data to copy
 * @skip:          Offset within the BO
 * Return:         0 on success or appropriate error number
 *
 * Copy contents of previously allocated device memory to host buffer. ``skip`` specifies how many bytes
 * to skip from the beginning of the BO before copying-out ``size`` bytes of device buffer.
 */
XCL_DRIVER_DLLESPEC size_t xclReadBO(xclDeviceHandle handle, unsigned int boHandle,
                                     void *dst, size_t size, size_t skip);

/**
 * xclMapBO() - Memory map BO into user's address space
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @write:         READ only or READ/WRITE mapping
 * Return:         Memory mapped buffer
 *
 * Map the contents of the buffer object into host memory
 * To unmap the buffer call POSIX unmap() on mapped void * pointer returned from xclMapBO
 */
XCL_DRIVER_DLLESPEC void *xclMapBO(xclDeviceHandle handle, unsigned int boHandle, bool write);

/**
 * xclSyncBO() - Synchronize buffer contents in requested direction
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @dir:           To device or from device
 * @size:          Size of data to synchronize
 * @offset:        Offset within the BO
 * Return:         0 on success or standard errno
 *
 * Synchronize the buffer contents between host and device. Depending on the memory model this may
 * require DMA to/from device or CPU cache flushing/invalidation
 */
XCL_DRIVER_DLLESPEC int xclSyncBO(xclDeviceHandle handle, unsigned int boHandle, enum xclBOSyncDirection dir,
                                  size_t size, size_t offset);
/**
 * xclCopyBO() - Copy device buffer contents to another buffer
 *
 * @handle:        Device handle
 * @dstBoHandle:   Destination BO handle
 * @srcBoHandle:   Source BO handle
 * @size:          Size of data to synchronize
 * @dst_offset:    dst  Offset within the BO
 * @src_offset:    src  Offset within the BO
 * Return:         0 on success or standard errno
 *
 * Copy from source buffer contents to destination buffer, can be device to device or device to host.
 * Always perform WRITE to achieve better performance, destination buffer can be on device or host
 * require DMA from device
 */
XCL_DRIVER_DLLESPEC int xclCopyBO(xclDeviceHandle handle, unsigned int dstBoHandle, unsigned int srcBoHandle,
                                   size_t size, size_t dst_offset, size_t src_offset);

/**
 * xclExportBO() - Obtain DMA-BUF file descriptor for a BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle which needs to be exported
 * Return:         File handle to the BO or standard errno
 *
 * Export a BO for import into another device or Linux subsystem which accepts DMA-BUF fd
 * This operation is backed by Linux DMA-BUF framework
 */
XCL_DRIVER_DLLESPEC int xclExportBO(xclDeviceHandle handle, unsigned int boHandle);

/**
 * xclImportBO() - Obtain BO handle for a BO represented by DMA-BUF file descriptor
 *
 * @handle:        Device handle
 * @fd:            File handle to foreign BO owned by another device which needs to be imported
 * @flags:         Unused
 * Return:         BO handle of the imported BO
 *
 * Import a BO exported by another device.     *
 * This operation is backed by Linux DMA-BUF framework
 */
XCL_DRIVER_DLLESPEC unsigned int xclImportBO(xclDeviceHandle handle, int fd, unsigned flags);

/**
 * xclGetBOProperties() - Obtain xclBOProperties struct for a BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @properties:    BO properties struct pointer
 * Return:         0 on success
 *
 * This is the prefered method for obtaining BO property information.
 */
XCL_DRIVER_DLLESPEC int xclGetBOProperties(xclDeviceHandle handle, unsigned int boHandle,
                                           struct xclBOProperties *properties);

/*
 * xclGetBOSize() - Retrieve size of a BO
 *
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * Return          size_t size of the BO on success
 *
 * This API is deprecated and will be removed in future release.
 * New clients should use xclGetBOProperties() instead
 */
inline XCL_DRIVER_DLLESPEC size_t xclGetBOSize(xclDeviceHandle handle, unsigned int boHandle)
{
    struct xclBOProperties p;
    return !xclGetBOProperties(handle, boHandle, &p) ? (size_t)p.size : -1;
}

/*
 * Get the physical address on the device
 *
 * This API is deprecated and will be removed in future release.
 * New clients should use xclGetBOProperties() instead.
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @return         uint64_t address of the BO on success
 */
inline XCL_DRIVER_DLLESPEC uint64_t xclGetDeviceAddr(xclDeviceHandle handle, unsigned int boHandle)
{
    struct xclBOProperties p;
    return !xclGetBOProperties(handle, boHandle, &p) ? p.paddr : -1;
}

/* End XRT Buffer Management APIs */

/**
 * DOC: XRT Unmanaged DMA APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Unmanaged DMA APIs are for exclusive use by the debuggers and tools. The APIs allow clinets to read/write
 * from/to absolute device address. No checks are performed if a buffer was allocated before at the specified
 * location or if the address is valid. Users who want to take over the full memory managemnt of the device
 * may use this API to synchronize their buffers between host and device.
 */

/**
 * xclUnmgdPread() - Perform unmanaged device memory read operation
 *
 * @handle:        Device handle
 * @flags:         Unused
 * @buf:           Destination data pointer
 * @size:          Size of data to copy
 * @offset:        Absolute offset inside device
 * Return:         0 on success or appropriate error number
 *
 * This API may be used to perform DMA operation from absolute location specified. Users
 * may use this if they want to perform their own device memory management -- not using the buffer
 * object (BO) framework defined before.
 */
XCL_DRIVER_DLLESPEC ssize_t xclUnmgdPread(xclDeviceHandle handle, unsigned flags, void *buf,
                                          size_t size, uint64_t offset);

/**
 * xclUnmgdPwrite() - Perform unmanaged device memory read operation
 *
 * @handle:        Device handle
 * @flags:         Unused
 * @buf:           Source data pointer
 * @size:          Size of data to copy
 * @offset:        Absolute offset inside device
 * Return:         0 on success or appropriate error number
 *
 * This API may be used to perform DMA operation to an absolute location specified. Users
 * may use this if they want to perform their own device memory management -- not using the buffer
 * object (BO) framework defined before.
 */
XCL_DRIVER_DLLESPEC ssize_t xclUnmgdPwrite(xclDeviceHandle handle, unsigned flags, const void *buf,
                                           size_t size, uint64_t offset);

/* End XRT Unmanaged DMA APIs */

/*
 * DOC: XRT Register read/write APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * These functions are used to read and write peripherals sitting on the address map.
 * Note that the offset is wrt the address space. The register map and address map of
 * execution units can be obtained from xclbin. Note that these APIs are multi-threading/
 * multi-process safe and no checks are performed on the read/write requests.
 * OpenCL runtime does **not** use these APIs but instead uses execution management APIs
 * defined below.
 */

/**
 * xclWrite() - Perform register write operation, deprecated
 *
 * @handle:        Device handle
 * @space:         Address space
 * @offset:        Offset in the address space
 * @hostBuf:       Source data pointer
 * @size:          Size of data to copy
 * Return:         size of bytes written or appropriate error number
 *
 * This API may be used to write to device registers exposed on PCIe BAR. Offset is relative to the
 * the address space. A device may have many address spaces.
 * *This API is deprecated. Please use xclRegWrite(), instead.*
 */

XCL_DRIVER_DLLESPEC size_t xclWrite(xclDeviceHandle handle, enum xclAddressSpace space, uint64_t offset,
                                    const void *hostBuf, size_t size) __attribute__ ((deprecated));

/**
 * xclRead() - Perform register read operation, deprecated
 *
 * @handle:        Device handle
 * @space:         Address space
 * @offset:        Offset in the address space
 * @hostbuf:       Destination data pointer
 * @size:          Size of data to copy
 * Return:         size of bytes written or appropriate error number
 *
 * This API may be used to read from device registers exposed on PCIe BAR. Offset is relative to the
 * the address space. A device may have many address spaces.
 * *This API is deprecated. Please use xclRegRead(), instead.*
 */
XCL_DRIVER_DLLESPEC size_t xclRead(xclDeviceHandle handle, enum xclAddressSpace space, uint64_t offset,
                                   void *hostbuf, size_t size) __attribute__ ((deprecated));

/* XRT Register read/write APIs */

/*
 * TODO:
 * Define the following APIs
 *
 * 1. Host accessible pipe APIs: pread/pwrite
 * 2. Accelerator status, start, stop APIs
 * 3. Context creation APIs to support multiple clients
 * 4. Multiple OCL Region support
 * 5. DPDK style buffer management and device polling
 *
 */

/**
 * DOC: XRT Compute Unit Execution Management APIs
 *
 * These APIs are under development. These functions will be used to start compute
 * units and wait for them to finish.
 */

/**
 * xclExecBuf() - Submit an execution request to the embedded (or software) scheduler
 *
 * @handle:        Device handle
 * @cmdBO:         BO handle containing command packet
 * Return:         0 or standard error number
 *
 * Submit an exec buffer for execution. The exec buffer layout is defined by struct ert_packet
 * which is defined in file *ert.h*. The BO should been allocated with DRM_XOCL_BO_EXECBUF flag.
 */
XCL_DRIVER_DLLESPEC int xclExecBuf(xclDeviceHandle handle, unsigned int cmdBO);

/**
 * xclExecBufWithWaitList() - Submit an execution request to the embedded (or software) scheduler
 *
 * @handle:              Device handle
 * @cmdBO:               BO handle containing command packet
 * @num_bo_in_wait_list: Number of BO handles in wait list
 * @bo_wait_list:        BO handles that must complete execution before cmdBO is started
 * Return:               0 or standard error number
 *
 * Submit an exec buffer for execution. The BO handles in the wait
 * list must complete execution before cmdBO is started.  The BO
 * handles in the wait list must have beeen submitted prior to this
 * call to xclExecBufWithWaitList.
 */
XCL_DRIVER_DLLESPEC int xclExecBufWithWaitList(xclDeviceHandle handle, unsigned int cmdBO,
                                               size_t num_bo_in_wait_list, unsigned int *bo_wait_list);

/**
 * xclExecWait() - Wait for one or more execution events on the device
 *
 * @handle:                  Device handle
 * @timeoutMilliSec:         How long to wait for
 * Return:                   Same code as poll system call
 *
 * Wait for notification from the hardware. The function essentially calls "poll" system
 * call on the driver file handle. The return value has same semantics as poll system call.
 * If return value is > 0 caller should check the status of submitted exec buffers
 * Note that if you perform wait for the same handle from multiple threads, you
 * may lose wakeup for some of them. So, use different handle in different threads.
 */
XCL_DRIVER_DLLESPEC int xclExecWait(xclDeviceHandle handle, int timeoutMilliSec);

/**
 * xclRegisterInterruptNotify() - register *eventfd* file handle for a MSIX interrupt
 *
 * @handle:        Device handle
 * @userInterrupt: MSIX interrupt number
 * @fd:            Eventfd handle
 * Return:         0 on success or standard errno
 *
 * Support for non managed interrupts (interrupts from custom IPs). fd should be obtained from
 * eventfd system call. Caller should use standard poll/read eventfd framework in order to wait for
 * interrupts. The handles are automatically unregistered on process exit.
 */
XCL_DRIVER_DLLESPEC int xclRegisterInterruptNotify(xclDeviceHandle handle, unsigned int userInterrupt,
                                                   int fd);

/* XRT Compute Unit Execution Management APIs */

/**
 * DOC: XRT Stream Queue APIs
 *
 * These functions are used for next generation DMA Engine, QDMA. QDMA provides not only memory
 * mapped DMA which moves data between host memory and board memory, but also stream DMA which moves
 * data between host memory and kernel directly. XDMA memory mapped DMA APIs are also supported on
 * QDMA. New stream APIs are provided here for preview and may be revised in a future release. These
 * can only be used with platforms with QDMA engine under the hood. The higher level OpenCL based
 * streaming APIs offer more refined interfaces and compatibility between releases and each stream maps
 * to a QDMA queue underneath.
 */

enum xclStreamContextFlags {
	/* Enum for xclQueueContext.flags */
	XRT_QUEUE_FLAG_POLLING		= (1 << 2),
};

/*
 * struct xclQueueContext - structure to describe a Queue
 */
struct xclQueueContext {
    uint32_t	type;	   /* stream or packet Queue, read or write Queue*/
    uint32_t	state;	   /* initialized, running */
    uint64_t	route;	   /* route id from xclbin */
    uint64_t	flow;	   /* flow id from xclbin */
    uint32_t	qsize;	   /* number of descriptors */
    uint32_t	desc_size; /* this might imply max inline msg size */
    uint64_t	flags;	   /* isr en, wb en, etc */
};

/**
 * xclCreateWriteQueue - Create Write Queue
 *
 * @handle:        Device handle
 * @q_ctx:         Queue Context
 * @q_hdl:         Queue handle
 * Return:         0 or appropriate error number
 *
 * Create write queue based on information provided in Queue context. Queue handle is generated if creation successes.
 */
XCL_DRIVER_DLLESPEC int xclCreateWriteQueue(xclDeviceHandle handle, struct xclQueueContext *q_ctx,  uint64_t *q_hdl);

/**
 * xclCreateReadQueue - Create Read Queue
 *
 * @handle:        Device handle
 * @q_ctx:         Queue Context
 * @q_hdl:         Queue handle
 * Return:         0 or appropriate error number
 *
 * Create read queue based on information provided in Queue context. Queue handle is generated if creation successes.
 */
XCL_DRIVER_DLLESPEC int xclCreateReadQueue(xclDeviceHandle handle, struct xclQueueContext *q_ctx, uint64_t *q_hdl);

/**
 * xclDestroyQueue - Destroy Queue
 *
 * @handle:        Device handle
 * @q_hdl:         Queue handle
 * Return:         0 or appropriate error number
 *
 * Destroy read or write queue and release all queue resources.
 */
XCL_DRIVER_DLLESPEC int xclDestroyQueue(xclDeviceHandle handle, uint64_t q_hdl);

/**
 * xclAllocQDMABuf - Allocate DMA buffer
 *
 * @handle:        Device handle
 * @size:          Buffer size
 * @buf_hdl:       Buffer handle
 * Return:         0 or appropriate error number
 *
 * Allocate DMA buffer which is used for queue read and write.
 */
XCL_DRIVER_DLLESPEC void *xclAllocQDMABuf(xclDeviceHandle handle, size_t size, uint64_t *buf_hdl);

/**
 * xclFreeQDMABuf - Free DMA buffer
 *
 * @handle:        Device handle
 * @buf_hdl:       Buffer handle
 * Return:         0 or appropriate error number
 *
 * Free DMA buffer allocated by xclAllocQDMABuf.
 */
XCL_DRIVER_DLLESPEC int xclFreeQDMABuf(xclDeviceHandle handle, uint64_t buf_hdl);

/*
 * xclModifyQueue - Modify Queue
 *
 * @handle:		Device handle
 * @q_hdl:		Queue handle
 *
 * This function modifies Queue context on the fly. Modifying rid implies
 * to program hardware traffic manager to connect Queue to the kernel pipe.
 */
XCL_DRIVER_DLLESPEC int xclModifyQueue(xclDeviceHandle handle, uint64_t q_hdl);

/*
 * xclStartQueue - set Queue to running state
 * @handle:             Device handle
 * @q_hdl:              Queue handle
 *
 * This function set xclStartQueue to running state. xclStartQueue starts to process Read and Write requests.
 * TODO: remove this
 */
XCL_DRIVER_DLLESPEC int xclStartQueue(xclDeviceHandle handle, uint64_t q_hdl);

/*
 * xclStopQueue - set Queue to init state
 * @handle:             Device handle
 * @q_hdl:              Queue handle
 *
 * This function set Queue to init state. all pending read and write requests will be flushed.
 * wr_complete and rd_complete will be called with error wbe for flushed requests.
 * TODO: remove this
 */
XCL_DRIVER_DLLESPEC int xclStopQueue(xclDeviceHandle handle, uint64_t q_hdl);

/*
 * struct xclWRBuffer
 */
struct xclReqBuffer {
    union {
	char*    buf;    // ptr or,
	uint64_t va;	 // offset
    };
    uint64_t  len;
    uint64_t  buf_hdl;   // NULL when first field is buffer pointer
};

/*
 * enum xclQueueRequestKind - request type.
 */
enum xclQueueRequestKind {
    XCL_QUEUE_WRITE = 0,
    XCL_QUEUE_READ  = 1,
    //More, in-line etc.
};

/*
 * enum xclQueueRequestFlag - flags associated with the request.
 */
/* this has to be the same with Xfer flags defined in opencl CL_STREAM* */
enum xclQueueRequestFlag {
    XCL_QUEUE_REQ_EOT			= 1 << 0,
    XCL_QUEUE_REQ_CDH			= 1 << 1,
    XCL_QUEUE_REQ_NONBLOCKING		= 1 << 2,
    XCL_QUEUE_REQ_SILENT		= 1 << 3, /* not supp. not generate event for non-blocking req */
};

/*
 * struct xclQueueRequest - read and write request
 */
struct xclQueueRequest {
    enum xclQueueRequestKind op_code;
    struct xclReqBuffer*       bufs;
    uint32_t	        buf_num;
    char*               cdh;
    uint32_t	        cdh_len;
    uint32_t		flag;
    void*		priv_data;
    uint32_t            timeout;
};

/*
 * struct xclReqCompletion - read/write completion
 * keep this in sync with cl_streams_poll_req_completions
 * in core/include/stream.h
 */
struct xclReqCompletion {
    char			resv[64]; /* reserved for meta data */
    void			*priv_data;
    size_t			nbytes;
    int				err_code;
};

/**
 * xclWriteQueue - write data to queue
 * @handle:        Device handle
 * @q_hdl:         Queue handle
 * @wr_req:        Queue request
 * Return:         Number of bytes been written or appropriate error number
 *
 * Move data from host memory to board. The destination is determined by flow id and route id which are provided to xclCreateWriteQueue. it returns number of bytes been moved or error code.
 * By default, this function returns only when the entire buf has been written, or error. If XCL_QUEUE_REQ_NONBLOCKING flag is used, it returns immediately and xclPollCompletion needs to be used to determine if the data transmission is completed.
 * If XCL_QUEUE_REQ_EOT flag is used, end of transmit signal will be added at the end of this tranmission.
 */
XCL_DRIVER_DLLESPEC ssize_t xclWriteQueue(xclDeviceHandle handle, uint64_t q_hdl, struct xclQueueRequest *wr_req);

/**
 * xclReadQueue - read data from queue
 * @handle:        Device handle
 * @q_hdl:         Queue handle
 * @rd_req:        read request
 * Return:         Number of bytes been read or appropriate error number
 *
 * Move data from board to host memory. The source is determined by flow id and route id which are provided to xclCreateReadQueue. It returns number of bytes been moved or error code.
 * This function returns until all the requested bytes is read or error happens. If XCL_QUEUE_REQ_NONBLOCKING flag is used, it returns immediately and xclPollCompletion needs to be used to determine if the data trasmission is completed.
 * If XCL_QUEUE_REQ_EOT flag is used, data transmission for the current read request completes immediatly once end of transmit signal is received.
 */
XCL_DRIVER_DLLESPEC ssize_t xclReadQueue(xclDeviceHandle handle, uint64_t q_hdl, struct xclQueueRequest *rd_req);

/**
 * xclPollCompletion - poll read/write queue completion
 * @min_compl:     Unblock only when receiving min_compl completions
 * @max_compl:     Max number of completion with one poll
 * @comps:         Completed request array
 * @actual_compl:  Number of requests been completed
 * @timeout:       Timeout
 * Return:         Number of events or appropriate error number
 *
 * Poll completion events of non-blocking read/write requests. Once this function returns, an array of completed requests is returned.
 */
XCL_DRIVER_DLLESPEC int xclPollCompletion(xclDeviceHandle handle, int min_compl, int max_compl, struct xclReqCompletion *comps, int* actual_compl, int timeout);

XCL_DRIVER_DLLESPEC const struct axlf_section_header* wrap_get_axlf_section(const struct axlf* top, enum axlf_section_kind kind);

XCL_DRIVER_DLLESPEC size_t xclDebugReadIPStatus(xclDeviceHandle handle, enum xclDebugReadType type,
                                                                           void* debugResults);


#ifdef __cplusplus
}
#endif

#endif

#include <xrt/xrt.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_aie.h>
#include <xrt/experimental/xrt_bo.h>
#include <xrt/experimental/xrt_device.h>

#define DEF_XCLBIN "/media/sd-mmcblk0p1/dpu.xclbin"
/**
 *  * xrtDeviceOpenFromXcl() - Open a device from a shim xclDeviceHandle
 *  *
 *  * @xhdl:         Shim xclDeviceHandle
 *  * Return:        Handle representing the opened device, or nullptr on error
 *  *
 *  * The returned XRT device handle must be explicitly closed when
 *  * nolonger needed.
 *  */

class vai_aie_task_handler {
    public:
        vai_aie_task_handler(xclDeviceHandle xhcl, const char* xclbin_path=DEF_XCLBIN, int dev_id=0) {
#if !defined(__AIESIM__)
            dhdl = nullptr;
            // Create XRT device handle for XRT API
            xclbinFilename = xclbin_path;
            dhdl = xrtDeviceOpenFromXcl(xhcl);
            xrtDeviceLoadXclbinFile(dhdl,xclbinFilename);
            xrtDeviceGetXclbinUUID(dhdl, uuid);
#endif
        };
        vai_aie_task_handler(const char* xclbin_path=DEF_XCLBIN, int dev_id=0) {
#if !defined(__AIESIM__)
            dhdl = nullptr;
            // Create XRT device handle for XRT API
            xclbinFilename = xclbin_path;
            dhdl = xrtDeviceOpen(dev_id);//device index=0
            xrtDeviceLoadXclbinFile(dhdl,xclbinFilename);
            xrtDeviceGetXclbinUUID(dhdl, uuid);
#endif
        };
        ~vai_aie_task_handler() {
#if !defined(__AIESIM__)
            xrtDeviceClose(dhdl);
#endif
        };

#if !defined(__AIESIM__)
        xrtDeviceHandle dhdl;
        xuid_t uuid;
#endif
    private:
#if !defined(__AIESIM__)
        const char* xclbinFilename;
#endif
};

extern int xrtSyncBOAIENB(xrtDeviceHandle handle, xrtBufferHandle bohdl, const char *gmioName, enum xclBOSyncDirection dir, size_t size, size_t offset);
extern int xrtGMIOWait(xrtDeviceHandle handle, const char *gmioName);

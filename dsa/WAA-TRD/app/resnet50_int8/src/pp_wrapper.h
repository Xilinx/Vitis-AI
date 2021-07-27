#include <sys/time.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

// host_src includes
#include "xclhal2.h"
#include "xclbin.h"

// lowlevel common include
#include "utils.h"

#define DSA64 1

#if defined(DSA64)
#include "xf_pppre_hw_64.h"
#else
#include "xf_pppre_hw.h"
#endif

class PPHandle
{
public:
	xclDeviceHandle handle;
	uint64_t cu_base_addr;
	unsigned cu_index;
	unsigned bo_paramsbuf_Handle;
	uint64_t bo_paramsbuf_devAddr;
	unsigned bo_img_inp_Handle;
	unsigned char *img_inptr;
	uint64_t bo_input_phy_addr;
	unsigned bo_img_out_Handle;
	int8_t *outptr;
	uint64_t bo_output_phy_addr;
	float *paramsbuf_ptr;
	unsigned execHandle;
	void *execData;
};

extern PPHandle *pphandle;

static std::vector<std::string> get_xclbins_in_dir(std::string path)
{
	if (path.find(".xclbin") != std::string::npos)
		return {path};

	std::vector<std::string> xclbinPaths;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(path.c_str())) != NULL)
	{
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL)
		{
			std::string name(ent->d_name);
			if (name.find(".xclbin") != std::string::npos)
				xclbinPaths.push_back(path + "/" + name);
		}
		closedir(dir);
	}
	return xclbinPaths;
}

int preprocess(PPHandle *&pphandle,
			   unsigned char *inimg_data,
			   int img_ht,
			   int img_wt,
			   int org_ht,
			   int org_wt,
			   int8_t *&out_data)
{
	xclDeviceHandle handle = pphandle->handle;
	unsigned bo_img_inp_Handle = pphandle->bo_img_inp_Handle;
	unsigned char *img_inptr = pphandle->img_inptr;
	uint64_t bo_input_phy_addr = pphandle->bo_input_phy_addr;

	unsigned bo_img_out_Handle = pphandle->bo_img_out_Handle;
	uint64_t bo_output_phy_addr = pphandle->bo_output_phy_addr;
	int8_t *outptr = pphandle->outptr;
	unsigned execHandle = pphandle->execHandle;
	void *execData = pphandle->execData;

	const int imageToDevice_size = img_wt * img_ht * 3 * sizeof(char);

	std::memcpy(img_inptr, inimg_data, imageToDevice_size);

	// Send the input imageToDevice data to the device memory
	//std::cout << "Send the imageToDevice input data to the device memory.\n";
	if (xclSyncBO(handle, bo_img_inp_Handle, XCL_BO_SYNC_BO_TO_DEVICE, imageToDevice_size, 0))
	{
		return 1;
	}

	auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd *>(execData);
	ecmd2->data[XFPPPre_CONTROL_ADDR_rows_in / 4] = img_ht;
	ecmd2->data[XFPPPre_CONTROL_ADDR_cols_in / 4] = img_wt;

	int ret;
	if ((ret = xclExecBuf(handle, execHandle)) != 0)
	{
		std::cout << "Unable to trigger Preprocess, error:" << ret << std::endl;
		return ret;
	}
	do
	{
		ret = xclExecWait(handle, 1000);
		if (ret == 0)
		{
			std::cout << "Preprocess Task Time out, state =" << ecmd2->state << "cu_mask = " << ecmd2->cu_mask << std::endl;
		}
		else if (ecmd2->state == ERT_CMD_STATE_COMPLETED)
		{
			break;
		}
	} while (1);
	if (xclUnmgdPread(handle, 0, out_data, org_ht * org_wt * 3, bo_output_phy_addr))
		throw std::runtime_error("Error: dump failed!");

	return 0;
}

void release_pp(PPHandle *&pphandle)
{
	xclDeviceHandle handle = pphandle->handle;
	xclFreeBO(handle, pphandle->bo_paramsbuf_Handle);
	xclFreeBO(handle, pphandle->bo_img_inp_Handle);
	xclFreeBO(handle, pphandle->bo_img_out_Handle);
	xclClose(handle);

	return;
}

int pp_kernel_init(PPHandle *&pphandle,
				   float *mean,
				   float input_scale,
				   int out_ht,
				   int out_wt)
{
	// get xclbin path and acquire handle
	const char *xclbinPath = std::getenv("XLNX_VART_FIRMWARE");

	if (xclbinPath == nullptr)
		throw std::runtime_error("Error: xclbinPath is not set, please consider setting XLNX_VART_FIRMWARE.");

	// get available xclbins
	auto xclbins = get_xclbins_in_dir(xclbinPath);
	const char *xclbin = xclbins[0].c_str();

	PPHandle *my_handle = new PPHandle;
	pphandle = my_handle = (PPHandle *)my_handle;

	unsigned index = 0;
	std::string halLogfile;
	unsigned cu_index = 2;

	xclDeviceHandle handle;
	uint64_t cu_base_addr = 0;
	uuid_t xclbinId;
	int first_mem = -1;
	bool ret_initXRT = 0;
	bool ret_firstmem = 0;
	bool ret_runkernel = 0;
	bool ret_checkDevMem = 0;

	if (initXRT(xclbin, index, halLogfile.c_str(), handle, cu_index, cu_base_addr, first_mem, xclbinId))
		ret_initXRT = 1;
	if (xclOpenContext(handle, xclbinId, cu_index, true))
		throw std::runtime_error("Cannot create context");

	float params[9];
	//# Mean params
	params[0] = mean[0];
	params[1] = mean[1];
	params[2] = mean[2];

	//# Input scale
	params[3] = params[4] = params[5] = input_scale;

	//# Set to default zero
	params[6] = params[7] = params[8] = 0.0;

	const int paramsbuf_size = 9 * sizeof(float);
	unsigned bo_paramsbuf_Handle = xclAllocBO(handle, paramsbuf_size, XCL_BO_DEVICE_RAM, 1);
	float *paramsbuf_ptr = (float *)xclMapBO(handle, bo_paramsbuf_Handle, true);

	if ((paramsbuf_ptr == NULL))
		throw std::runtime_error("paramsbuf pointer is invalid\n");

	std::memcpy(paramsbuf_ptr, params, paramsbuf_size);

	// Get & check the device memory address
	xclBOProperties p;
	uint64_t bo_paramsbuf_devAddr = !xclGetBOProperties(handle, bo_paramsbuf_Handle, &p) ? p.paddr : -1;
	if ((bo_paramsbuf_devAddr == (uint64_t)(-1)))
	{
		ret_checkDevMem = 1;
	}

	if (xclSyncBO(handle, bo_paramsbuf_Handle, XCL_BO_SYNC_BO_TO_DEVICE, paramsbuf_size, 0))
	{
		return 1;
	}

	//# creating memory for 4K image
	const int imageToDevice_size = 3840 * 2160 * 3 * sizeof(char);

	// Allocate the device memory
	unsigned bo_img_inp_Handle = xclAllocBO(handle, imageToDevice_size, XCL_BO_DEVICE_RAM, 1);

	// Create the mapping to the host memory
	unsigned char *img_inptr = (unsigned char *)xclMapBO(handle, bo_img_inp_Handle, true);

	if ((img_inptr == NULL))
		throw std::runtime_error("imageToDevice pointer is invalid\n");

	uint64_t bo_input_phy_addr = !xclGetBOProperties(handle, bo_img_inp_Handle, &p) ? p.paddr : -1;
	if ((bo_input_phy_addr == (uint64_t)(-1)))
	{
		ret_checkDevMem = 1;
	}

	const int outimg_size = out_ht * out_wt * 3 * sizeof(char);
	// Allocate the device memory
	unsigned bo_img_out_Handle = xclAllocBO(handle, outimg_size, XCL_BO_DEVICE_RAM, 1);
	// Create the mapping to the host memory
	int8_t *outptr = (int8_t *)xclMapBO(handle, bo_img_out_Handle, false);
	if ((outptr == NULL))
		throw std::runtime_error("outptr pointer is invalid\n");
	uint64_t bo_output_phy_addr = !xclGetBOProperties(handle, bo_img_out_Handle, &p) ? p.paddr : -1;
	if ((bo_output_phy_addr == (uint64_t)(-1)))
	{
		ret_checkDevMem = 1;
	}

	//thread_local static
	unsigned execHandle = 0;
	//thread_local static
	void *execData = nullptr;

	int th1 = 127, th2 = 128;

	if (execHandle == 0)
		execHandle = xclAllocBO(handle, 4096, xclBOKind(0), (1 << 31));
	if (execData == nullptr)
		execData = xclMapBO(handle, execHandle, true);

	auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd *>(execData);

	// Clear the command in case it was recycled
	size_t regmap_size = XFPPPre_CONTROL_ADDR_th2 / 4 + 1; // regmap
	std::memset(ecmd2, 0, (sizeof *ecmd2) + regmap_size);

	// Program the command packet header
	ecmd2->state = ERT_CMD_STATE_NEW;
	ecmd2->opcode = ERT_START_CU;
	ecmd2->count = 1 + regmap_size; // cu_mask + regmap

	// Program the CU mask. One CU at index 0
	ecmd2->cu_mask = 0x4;

	// Program the register map
	ecmd2->data[XFPPPre_CONTROL_ADDR_AP_CTRL] = 0x0; // ap_start

#if defined(DSA64)
	//std::cout << "64-bit address\n";
	ecmd2->data[XFPPPre_CONTROL_ADDR_img_inp / 4] = bo_input_phy_addr & 0xFFFFFFFF;
	ecmd2->data[XFPPPre_CONTROL_ADDR_img_inp / 4 + 1] = (bo_input_phy_addr >> 32) & 0xFFFFFFFF;
	ecmd2->data[XFPPPre_CONTROL_ADDR_img_out / 4] = bo_output_phy_addr & 0xFFFFFFFF;
	ecmd2->data[XFPPPre_CONTROL_ADDR_img_out / 4 + 1] = (bo_output_phy_addr >> 32) & 0xFFFFFFFF;
	ecmd2->data[XFPPPre_CONTROL_ADDR_params / 4] = bo_paramsbuf_devAddr & 0xFFFFFFFF;
	ecmd2->data[XFPPPre_CONTROL_ADDR_params / 4 + 1] = (bo_paramsbuf_devAddr >> 32) & 0xFFFFFFFF;
#else
	//std::cout << "32-bit address\n";
	ecmd2->data[XFPPPre_CONTROL_ADDR_img_inp / 4] = bo_input_phy_addr;
	ecmd2->data[XFPPPre_CONTROL_ADDR_img_out / 4] = bo_output_phy_addr;
	ecmd2->data[XFPPPre_CONTROL_ADDR_params / 4] = bo_paramsbuf_devAddr;
#endif

	ecmd2->data[XFPPPre_CONTROL_ADDR_th1 / 4] = th1;
	ecmd2->data[XFPPPre_CONTROL_ADDR_th2 / 4] = th2;
	ecmd2->data[XFPPPre_CONTROL_ADDR_rows_out_resize / 4] = out_ht;
	ecmd2->data[XFPPPre_CONTROL_ADDR_cols_out_resize / 4] = out_wt;
	ecmd2->data[XFPPPre_CONTROL_ADDR_rows_out / 4] = out_ht;
	ecmd2->data[XFPPPre_CONTROL_ADDR_cols_out / 4] = out_wt;

	my_handle->handle = handle;
	my_handle->cu_base_addr = cu_base_addr;
	my_handle->cu_index = cu_index;
	my_handle->bo_paramsbuf_Handle = bo_paramsbuf_Handle;
	my_handle->bo_paramsbuf_devAddr = bo_paramsbuf_devAddr;
	my_handle->paramsbuf_ptr = paramsbuf_ptr;
	my_handle->bo_img_inp_Handle = bo_img_inp_Handle;
	my_handle->img_inptr = img_inptr;
	my_handle->bo_input_phy_addr = bo_input_phy_addr;
	my_handle->bo_img_out_Handle = bo_img_out_Handle;
	my_handle->outptr = outptr;
	my_handle->bo_output_phy_addr = bo_output_phy_addr;
	my_handle->execHandle = execHandle;
	my_handle->execData = execData;

	return 0;
}

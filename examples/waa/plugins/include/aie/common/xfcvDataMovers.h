/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef _XFCVDATAMOVERS_
#define _XFCVDATAMOVERS_

#include <adf/adf_api/XRTConfig.h>
#include <array>
#include <algorithm>
#include <common/smartTilerStitcher.hpp>
#include <experimental/xrt_kernel.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <vector>
#include <common/xf_aie_const.hpp>
#include <common/xf_aie_utils.hpp>

int xrtSyncBOAIENB(xrtDeviceHandle handle,
                   xrtBufferHandle bohdl,
                   const char* gmioName,
                   enum xclBOSyncDirection dir,
                   size_t size,
                   size_t offset);
int xrtGMIOWait(xrtDeviceHandle handle, const char* gmioName);

int xrtRunSetArgV(xrtRunHandle runHandle, int index, const void* value, size_t bytes);


namespace xF {

enum DataMoverKind { TILER, STITCHER };

template <int BITWIDTH>
class EmulAxiData {
    static constexpr int BYTEWIDTH = BITWIDTH / 8;

   public:
    char data[BYTEWIDTH];
    template <typename T>
    EmulAxiData(T m) {
        assert(sizeof(T) <= BYTEWIDTH);
        char* tmp = (char*)&m;
        for (unsigned int i = 0; i < BYTEWIDTH; i++) {
            data[i] = (i < sizeof(T)) ? tmp[i] : 0;
        }
    }
    template <typename T>
    EmulAxiData& operator=(const EmulAxiData& mc) {
        if (this != &mc) {
            for (unsigned int i = 0; i < BYTEWIDTH; i++) {
                data[i] = mc.data[i];
            }
        }
        return *this;
    }
};

template <typename T>
class CtypeToCVMatType {
   public:
    static constexpr uchar type =
        (std::is_same<T, float>::value)
            ? CV_32F
            : (std::is_same<T, double>::value)
                  ? CV_64F
                  : (std::is_same<T, int32_t>::value)
                        ? CV_32S
                        : (std::is_same<T, int16_t>::value)
                              ? CV_16S
                              : (std::is_same<T, uint16_t>::value)
                                    ? CV_16U
                                    : (std::is_same<T, int8_t>::value)
                                          ? CV_8S
                                          : (std::is_same<T, uint8_t>::value)
                                                ? CV_8U
                                                : (std::is_same<T, signed char>::value) ? CV_8S : CV_8U;
};

static xrtDeviceHandle gpDhdl = nullptr;
static std::vector<char> gHeader;
static const axlf* gpTop = nullptr;
static uint16_t gnTilerInstCount = 0;
static uint16_t gnStitcherInstCount = 0;


void deviceInit(const char* xclBin) {
    if (xclBin != nullptr) {
        if (gpDhdl == nullptr) {
            assert(gpTop == nullptr);

            gpDhdl = xrtDeviceOpen(0);
            if (gpDhdl == nullptr) {
                throw std::runtime_error("No valid device handle found. Make sure using right xclOpen index.");
            }

            std::ifstream stream(xclBin);
            stream.seekg(0, stream.end);
            size_t size = stream.tellg();
            stream.seekg(0, stream.beg);

            gHeader.resize(size);
            stream.read(gHeader.data(), size);

            gpTop = reinterpret_cast<const axlf*>(gHeader.data());
            if (xrtDeviceLoadXclbin(gpDhdl, gpTop)) {
                throw std::runtime_error("Xclbin loading failed");
            }

            adf::registerXRT(gpDhdl, gpTop->m_header.uuid);
        }
    }

    if (gpDhdl == nullptr) {
        throw std::runtime_error("No valid device handle found. Make sure using right xclOpen index.");
    }

    if (gpTop == nullptr) {
        throw std::runtime_error("Xclbin loading failed");
    }
}

struct xfcvDataMoverParams {
  cv::Size mInputImgSize;
  cv::Size mOutputImgSize;

  xfcvDataMoverParams() : mInputImgSize(0,0),
                          mOutputImgSize(0,0)
  {}

  xfcvDataMoverParams(const cv::Size &inImgSize)
    : mInputImgSize(inImgSize),
      mOutputImgSize(inImgSize)
  {}

  xfcvDataMoverParams(const cv::Size &inImgSize, const cv::Size &outImgSize)
    : mInputImgSize(inImgSize),
      mOutputImgSize(outImgSize)
  {}
};


template <DataMoverKind KIND,
          typename DATA_TYPE,
          int TILE_HEIGHT_MAX,
          int TILE_WIDTH_MAX,
          int AIE_VECTORIZATION_FACTOR,
          int CORES = 1,
          int PL_AXI_BITWIDTH = 16,
          bool USE_GMIO = false>
class xfcvDataMovers {
   private:
    uint16_t mOverlapH;
    uint16_t mOverlapV;
    uint16_t mTileRowsPerCore;
    uint16_t mTileColsPerCore;
    uint16_t mTileRows;
    uint16_t mTileCols;
    bool mbUserHndl;
    uint16_t mBurstLength;
    xrt::run runner;
    uint64_t dpu_phy_addr;
    cv::Mat* mpImage;
    std::array<uint16_t, 3> mImageSize;

    std::vector<smartTileMetaData> mMetaDataList;
    std::vector<EmulAxiData<PL_AXI_BITWIDTH> > mMetaDataVec;

    std::array<int, CORES> mMetadataSize;
    std::array<xrtBufferHandle, CORES> mMetadataBOHndl;
    xrtBufferHandle mImageBOHndl;

    std::array<xrtKernelHandle, CORES> mPLKHandleArr;
   
    std::array<xrtRunHandle, CORES> mPLRHandleArr;
   

    int imgSize() { return (mImageSize[0] * mImageSize[1] * mImageSize[2]); }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    int metaDataSizePerTile() {
        return (mMetaDataVec.size() * sizeof(EmulAxiData<PL_AXI_BITWIDTH>)) / mMetaDataList.size();
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    int metaDataSizePerTile() {
        return 0;
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    int metadataSize(int core) {
        return mMetadataSize[core];
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    int metadataSize(int core) {
        return 0;
    }

    auto metadataSize() { return mMetadataSize; }

    // Tiler copy {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void copy() {
        // Pack meta-data and image buffer in device buffer handle
        if (mTileColsPerCore == mTileCols) {
            // No column wise partition
            char* MetaDataVecP = (char*)mMetaDataVec.data();
            for (int i = 0; i < CORES; i++) {
                assert(mMetadataBOHndl[i]);
                char* metadata_buffer = (char*)xrtBOMap(mMetadataBOHndl[i]);
                memcpy(metadata_buffer, MetaDataVecP, metadataSize(i));
                MetaDataVecP += metadataSize(i);
            }
        } else {
            // Column wise partitioning
            int r = 0;
            int c = 0;
            for (int i = 0; i < CORES; i++) {
                assert(mMetadataBOHndl[i]);
                char* metadata_buffer = (char*)xrtBOMap(mMetadataBOHndl[i]);
                char* MetaDataVecP = ((char*)mMetaDataVec.data()) + (r * mTileCols + c) * metaDataSizePerTile();

                int x_r = tileRowsPerCore(i);
                int x_c = tileColsPerCore(i);

                int sz = x_c * metaDataSizePerTile();
                int stride = mTileCols * metaDataSizePerTile();
                for (int j = 0; j < x_r; j++) {
                    memcpy(metadata_buffer, MetaDataVecP, sz);
                    metadata_buffer += sz;
                    MetaDataVecP += stride;
                }

                c = c + mTileColsPerCore;
                if (c >= mTileCols) {
                    c = c - mTileCols;
                    r = (r + mTileRowsPerCore);
                }
            }
        }

        if (mbUserHndl == false) {
            assert(mpImage);
            assert(mImageBOHndl);
            void* buffer = xrtBOMap(mImageBOHndl);
            memcpy(buffer, mpImage->data, imgSize());
        }
    }
    //}

    // Stitcher copy {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void copy() {
        // No meta-data
        assert(mImageBOHndl);

        void* buffer = xrtBOMap(mImageBOHndl);
        if (mbUserHndl == false) {
            assert(mpImage);
            memcpy(mpImage->data, buffer, imgSize());
        } else {
            xrtBOSync(mImageBOHndl, XCL_BO_SYNC_BO_TO_DEVICE, imgSize(), 0);
        }
    }
    //}

    void free_metadata_buffer(int core) {
        if (mMetadataBOHndl[core] != nullptr) {
            xrtBOFree(mMetadataBOHndl[core]);
        }
        mMetadataBOHndl[core] = nullptr;
        mMetadataSize[core] = 0;
    }

    void free_metadata_buffer() {
        for (int i = 0; i < CORES; i++) {
            free_metadata_buffer(i);
        }
    }

    void alloc_metadata_buffer() {
        for (int i = 0; i < CORES; i++) {
            if (mMetadataBOHndl[i] == nullptr) {
                // Update size
                mMetadataSize[i] = metaDataSizePerTile() * tilesPerCore(i);

                // Allocate buffer
                // assert(metadataSize(i) > 0);
                //std::cout << "Allocating metadata device buffer (Tiler), "
                //          << " Size : " << metadataSize(i) << " bytes" << std::endl;
                //mMetadataBOHndl[i] = xrtBOAlloc(gpDhdl, metadataSize(i), 0, 0);
                mMetadataBOHndl[i] = xrtBOAlloc(gpDhdl, metadataSize(i), 0, xrtKernelArgGroupId(mPLKHandleArr[i],3));
            }
        }
    }

    void free_buffer() {
        if (mbUserHndl == false) {
            if (mImageBOHndl != nullptr) {
                xrtBOFree(mImageBOHndl);
            }
            mImageBOHndl = nullptr;
        }
    }

    void alloc_buffer() {
        if (mImageBOHndl == nullptr) {
            assert(imgSize() > 0);
            std::cout << "Allocating image device buffer (Tiler), "
                      << " Size : " << imgSize() << " bytes" << std::endl;
           // mImageBOHndl = xrtBOAlloc(gpDhdl, imgSize(), 0, 0);
            mImageBOHndl = xrtBOAlloc(gpDhdl, imgSize(), 0,  xrtKernelArgGroupId(mPLKHandleArr[0],2));
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    std::string krnl_inst_name(int n) {
        std::ostringstream ss;
        ss << "Tiler_top:{Tiler_top_" << n << "}";
        return ss.str();
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    std::string krnl_inst_name(int n) {
        std::ostringstream ss;
        ss << "stitcher_top:{stitcher_top_" << n << "}";
        return ss.str();
    }

   void load_krnl() {
        for (int i = 0; i < CORES; i++) {
            std::string name =
                (KIND == TILER) ? krnl_inst_name(++gnTilerInstCount) : krnl_inst_name(++gnStitcherInstCount);
            std::cout << "Loading kernel " << name.c_str() << std::endl;
           
            mPLKHandleArr[i] = xrtPLKernelOpen(gpDhdl, gpTop->m_header.uuid, name.c_str());
            mPLRHandleArr[i] = xrtRunOpen(mPLKHandleArr[i]);
            
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void setArgs() {
        //std::cout << "Setting kernel args (Tiler) ..." << std::endl;
        for (int i = 0; i < CORES; i++) {
            uint16_t r = tileRowsPerCore(i);
            uint16_t c = tileColsPerCore(i);
            uint32_t NumTiles = r*c;
            (void)xrtRunSetArg(mPLRHandleArr[i], 0, mImageSize[1]); // Stride is same as Width
            (void)xrtRunSetArg(mPLRHandleArr[i], 1, NumTiles); // Number of Tiles
            (void)xrtRunSetArg(mPLRHandleArr[i], 2, mImageBOHndl);
            (void)xrtRunSetArg(mPLRHandleArr[i], 3, mMetadataBOHndl[i]);
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void setArgs() {
        //std::cout << "Setting kernel args (Stitcher) ..." << std::endl;
        for (int i = 0; i < CORES; i++) {
            uint16_t r = tileRowsPerCore(i);
            uint16_t c = tileColsPerCore(i);
            uint32_t NumTiles = r*c;
            xrtRunSetArgV(mPLRHandleArr[i], 0, &mImageSize[1], 2); //Width
            xrtRunSetArgV(mPLRHandleArr[i], 1, &NumTiles, 4); // Number of Tiles
            xrtRunSetArgV(mPLRHandleArr[i], 2, &dpu_phy_addr, 8); // physical addr
            xrtRunSetArgV(mPLRHandleArr[i], 3, &r, 2); // Number of Tiles rows
            xrtRunSetArgV(mPLRHandleArr[i], 4, &c, 2); // Number of Tiles cols
        }
    }

   public:
    void start() {
        for (auto& r : mPLRHandleArr) {
            xrtRunStart(r);
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    xfcvDataMovers(uint16_t overlapH, uint16_t overlapV, int burst = 1) {
        if (gpDhdl == nullptr) {
            throw std::runtime_error("No valid device handle found. Make sure using xF::deviceInit(...) is called.");
        }

        mpImage = nullptr;
        mImageSize = {0, 0, 0};

        // Initialize overlaps
        mOverlapH = overlapH;
        mOverlapV = overlapV;

        mBurstLength = burst;

        mTileRowsPerCore = 0;
        mTileColsPerCore = 0;
        mTileRows = 0;
        mTileCols = 0;
        mbUserHndl = false;

        for (int i = 0; i < CORES; i++) {
            mMetadataBOHndl[i] = nullptr;
            mMetadataSize[i] = 0;
        }
        mImageBOHndl = nullptr;

        // Load the PL kernel
        load_krnl();
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    xfcvDataMovers(int burst = 1) {
        if (gpDhdl == nullptr) {
            throw std::runtime_error("No valid device handle found. Make sure using xF::deviceInit(...) is called.");
        }

        mpImage = nullptr;
        mImageSize = {0, 0, 0};

        // Initialize overlaps
        mOverlapH = 0;
        mOverlapV = 0;

        mBurstLength = burst;

        mTileRowsPerCore = 0;
        mTileColsPerCore = 0;
        mTileRows = 0;
        mTileCols = 0;
        mbUserHndl = false;

        for (int i = 0; i < CORES; i++) {
            mMetadataBOHndl[i] = nullptr;
            mMetadataSize[i] = 0;
        }
        mImageBOHndl = nullptr;

        // Load the PL kernel
        load_krnl();
    }

    // Non copyable {
    xfcvDataMovers(const xfcvDataMovers&) = delete;
    xfcvDataMovers& operator=(const xfcvDataMovers&) = delete;
    //}

    // Close / free operations tp be done here {
    ~xfcvDataMovers() {
        free_buffer();
        free_metadata_buffer();

        for (auto& r : mPLRHandleArr) {
            xrtRunClose(r);
        }
        
        for (auto& r : mPLKHandleArr) {
            xrtKernelClose(r);
        }
        if (gpDhdl != nullptr) {
            xrtDeviceClose(gpDhdl);
            gpDhdl = nullptr;
        }
    }
    //}

    void compute_metadata(const cv::Size& inImgSize, const cv::Size& outImgSize = cv::Size(0,0));

    // These functions will start the data transfer protocol {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    auto host2aie_nb(cv::Mat& img, xrtBufferHandle imgHndl = nullptr, const xfcvDataMoverParams& params = xfcvDataMoverParams()) {
        assert(sizeof(DATA_TYPE) >= img.elemSize());

        auto old_metadata_buffer_size = metadataSize();
        int old_img_buffer_size = imgSize();

        bool bRecompute = false;
        if ((mImageSize[0] != img.rows) || (mImageSize[1] != img.cols)) {
            bRecompute = true;
        }

        mpImage = &img;
        mImageSize = {(uint16_t)img.rows, (uint16_t)img.cols, (uint16_t)img.elemSize()};
        if (bRecompute == true) {
            // Pack metadata
            compute_metadata(img.size());
        }

        auto new_metadata_buffer_size = metadataSize();
        int new_img_buffer_size = imgSize();

        for (int i = 0; i < CORES; i++) {
            if (new_metadata_buffer_size[i] > old_metadata_buffer_size[i]) {
                free_metadata_buffer(i);
            }
        }

        if ((new_img_buffer_size > old_img_buffer_size) || (imgHndl != nullptr)) {
            free_buffer();
        }

        mbUserHndl = (imgHndl != nullptr);
        if (mbUserHndl) mImageBOHndl = imgHndl;

        // Allocate buffer
        alloc_metadata_buffer();
        alloc_buffer();

        // Copy input data to device buffer
        copy();

        // Set args
        setArgs();

        // Start the kernel
        start();

        std::array<uint16_t, 4> ret = {mTileRowsPerCore, mTileColsPerCore, mTileRows, mTileCols};
        return ret;
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    auto host2aie_nb(xrtBufferHandle imgHndl, const cv::Size& size, const xfcvDataMoverParams& params = xfcvDataMoverParams()) {
        cv::Mat img(size, CV_8UC1); // This image is redundant in case a handle is passed
        return host2aie_nb(img, imgHndl, params);
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void aie2host_nb(cv::Mat& img, std::array<uint16_t, 4> tiles, uint64_t dpu_input_phy_addr) {
        assert(sizeof(DATA_TYPE) >= img.elemSize());

        int old_img_buffer_size = imgSize();

        mpImage = &img;
        mImageSize = {(uint16_t)img.rows, (uint16_t)img.cols, (uint16_t)img.elemSize()};
        mTileRowsPerCore = tiles[0];
        mTileColsPerCore = tiles[1];
        mTileRows = tiles[2];
        mTileCols = tiles[3];

        int new_img_buffer_size = imgSize();
     
        dpu_phy_addr =  dpu_input_phy_addr ;  
       
        // Set args
        setArgs();

        // Start the kernel
        start();
         
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void aie2host_nb(xrtBufferHandle imgHndl, const cv::Size& size, std::array<uint16_t, 4> tiles) 
    {
        cv::Mat img(size, CV_8UC1); // This image is redundant in case a handle is passed
      
        uint64_t bo_phy_addr = xrtBOAddress(imgHndl);
        
        aie2host_nb(img, tiles, bo_phy_addr);
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void aie2host_nb(uint64_t dpu_input_phy_addr, const cv::Size& size, std::array<uint16_t, 4> tiles) {
        cv::Mat img(size, CV_8UC1); // This image is redundant in case a handle is passed
        aie2host_nb(img, tiles, dpu_input_phy_addr);
    }
    

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void wait() {
        for (auto& r : mPLRHandleArr) {
            (void)xrtRunWait(r);
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void wait() {
        for (auto& r : mPLRHandleArr) {
            (void)xrtRunWait(r);
        }

        // Copy data from device buffer to host
      //  copy();
    }

    uint16_t tilesPerCore() {
        if ((mTileRowsPerCore * mTileColsPerCore * CORES) != (mTileRows * mTileCols)) {
            std::cerr << "ERR: Tile rows distribution is not even across cores. Total number of generated tiles for "
                         "given resolution image and requested tile size is "
                      << "Rows(" << mTileRows << ") x Cols(" << mTileCols << ") and number of cores is " << CORES
                      << ". Please use core specific tile count function by passing core index of the corresponding "
                         "core as arguement value (eg. tilesPerCore(<core index>))."
                      << std::endl;
            exit(-1);
        }

        return (mTileRowsPerCore * mTileColsPerCore);
    }

    uint16_t tileColsPerCore(int core) {
        int hcore_distribution_factor = (mTileCols + mTileColsPerCore - 1) / mTileColsPerCore;
        int hidx = core % hcore_distribution_factor;

        uint16_t c_s = hidx * mTileColsPerCore;
        uint16_t c_e = std::min(mTileCols, uint16_t(c_s + mTileColsPerCore));

        return (c_e - c_s);
    }

    uint16_t tileRowsPerCore(int core) {
        int hcore_distribution_factor = (mTileCols + mTileColsPerCore - 1) / mTileColsPerCore;
        int vidx = core / hcore_distribution_factor;

        uint16_t r_s = vidx * mTileRowsPerCore;
        uint16_t r_e = std::min(mTileRows, uint16_t(r_s + mTileRowsPerCore));

        return (r_e - r_s);
    }

    uint16_t tilesPerCore(int core) {
        if (core >= CORES) {
            std::cerr << "ERR: Out of bound access. Trying to access " << core << " core in a " << CORES << " design."
                      << std::endl;
            exit(-1);
        }

        return tileRowsPerCore(core) * tileColsPerCore(core);
    }
};

template <DataMoverKind KIND,
          typename DATA_TYPE,
          int TILE_HEIGHT_MAX,
          int TILE_WIDTH_MAX,
          int AIE_VECTORIZATION_FACTOR,
          int CORES,
          int PL_AXI_BITWIDTH,
          bool USE_GMIO>
void xfcvDataMovers<KIND,
                    DATA_TYPE,
                    TILE_HEIGHT_MAX,
                    TILE_WIDTH_MAX,
                    AIE_VECTORIZATION_FACTOR,
                    CORES,
                    PL_AXI_BITWIDTH,
                    USE_GMIO>::compute_metadata(const cv::Size& inImgSize, const cv::Size& outImgSize) {
    mMetaDataList.clear();
    mMetaDataVec.clear();

    cv::Size inputImgSize = inImgSize;
    cv::Size outputImgSize = outImgSize;

    // Set default output image size to input image size
    if (outputImgSize == cv::Size(0,0)) {
      outputImgSize = inputImgSize;
    }

    mImageSize[0] = (uint16_t)inputImgSize.height;
    mImageSize[1] = (uint16_t)inputImgSize.width;

    //std::cout << "inputImgSize.height: " << inputImgSize.height << std::endl;
    //std::cout << "inputImgSize.width: " << inputImgSize.width << std::endl;
    //std::cout << "outputImgSize.height: " << outputImgSize.height << std::endl;
    //std::cout << "outputImgSize.width: " << outputImgSize.width << std::endl;
    //std::cout << "TileHeight: " << TILE_HEIGHT_MAX << std::endl;
    //std::cout << "TileWidth: " << TILE_WIDTH_MAX << std::endl;
    //std::cout << "OverlapH: " << mOverlapH << std::endl;
    //std::cout << "OverlapV: " << mOverlapV << std::endl;

    bool isOutputResize = false;
    if (inputImgSize.height == outputImgSize.height && inputImgSize.width == outputImgSize.width) {
        smartTileTilerGenerateMetaDataWithSpecifiedTileSize(
            {inputImgSize.height, inputImgSize.width}, mMetaDataList, mTileRows, mTileCols, {TILE_HEIGHT_MAX, TILE_WIDTH_MAX},
            {mOverlapH, mOverlapH}, {mOverlapV, mOverlapV}, AIE_VECTORIZATION_FACTOR, true);
    } else {
        isOutputResize = true;
        smartTileTilerGenerateMetaDataWithSpecifiedTileSize(
            {inputImgSize.height, inputImgSize.width}, {outputImgSize.height, outputImgSize.width}, mMetaDataList, mTileRows, mTileCols,
            AIE_VECTORIZATION_FACTOR, true);
    }

    char sMesg[2048];
    sMesg[0] = '\0';
    //sprintf(sMesg, "Requested tile size (%d,%d). Computed tile size (%d,%d). Number of tiles (%d,%d)\n",
    //        TILE_HEIGHT_MAX, TILE_WIDTH_MAX, mMetaDataList[0].tileHeight(), mMetaDataList[0].tileWidth(), mTileRows,
    //        mTileCols);
    std::cout << sMesg << std::endl;

    mTileRowsPerCore = (mTileRows + CORES - 1) / CORES;
    mTileColsPerCore = mTileCols;

    int OutOffset, InOffset, OutpositionV, OutpositionH, OutWidth, OutHeight;
    int InputImageStride = (int)inputImgSize.width;
    int OutputImageStride = (int)outputImgSize.width;
    int i=0;

    for (auto& metaData : mMetaDataList) {
        //std::cout << "position: (" << metaData.positionH() << "," << metaData.positionV() << ") "
        //          << "overlapH: (" << metaData.overlapSizeH_left() << "," << metaData.overlapSizeH_right() << ") "
        //          << "overlapV: (" << metaData.overlapSizeV_top() << "," << metaData.overlapSizeV_bottom() << ") "
        //          << "tilesize: (" << metaData.tileWidth() << "," << metaData.tileHeight() << ") "
        //          << std::endl;
        if (isOutputResize == false)
        {
            OutpositionV = metaData.positionV() + metaData.overlapSizeV_top();
            OutpositionH = metaData.positionH() + metaData.overlapSizeH_left();
            OutWidth  = (metaData.tileWidth()  - (metaData.overlapSizeH_left() + metaData.overlapSizeH_right()));
            OutHeight = (metaData.tileHeight() - (metaData.overlapSizeV_top()  + metaData.overlapSizeV_bottom()));
        }
        else
        {
            OutpositionV = i;
            OutpositionH = 0;
            OutWidth  = OutputImageStride;
            OutHeight = 1;
            i++;
        }
        OutOffset = ((        OutpositionV * OutputImageStride) + OutpositionH);
        InOffset  = ((metaData.positionV() * InputImageStride) + metaData.positionH());
        
        mMetaDataVec.emplace_back((int16_t)(InOffset & 0x0000ffff));  
        mMetaDataVec.emplace_back((int16_t)(InOffset >> 16)); 
        mMetaDataVec.emplace_back((int16_t)metaData.tileWidth());
        mMetaDataVec.emplace_back((int16_t)metaData.tileHeight());
        mMetaDataVec.emplace_back((int16_t)metaData.overlapSizeH_left());
        mMetaDataVec.emplace_back((int16_t)metaData.overlapSizeH_right());
        mMetaDataVec.emplace_back((int16_t)metaData.overlapSizeV_top());
        mMetaDataVec.emplace_back((int16_t)metaData.overlapSizeV_bottom());
        mMetaDataVec.emplace_back((int16_t)(OutOffset & 0x0000ffff));  
        mMetaDataVec.emplace_back((int16_t)(OutOffset >> 16)); 
        mMetaDataVec.emplace_back((int16_t)OutWidth);
        mMetaDataVec.emplace_back((int16_t)OutHeight);
        mMetaDataVec.emplace_back((int16_t)0x0050);  // Enable saturation, 1: 8U, 2: 8S
        mMetaDataVec.emplace_back((int16_t)OutpositionH);
        mMetaDataVec.emplace_back((int16_t)OutpositionV);
        mMetaDataVec.emplace_back((int16_t)metaData.positionH()); // In PosH
        mMetaDataVec.emplace_back((int16_t)metaData.positionV()); // In PosV
        mMetaDataVec.emplace_back((int16_t)16); // BIT_WIDTH
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);  
        mMetaDataVec.emplace_back((int16_t)0);
    }
}

template <DataMoverKind KIND,
          typename DATA_TYPE,
          int TILE_HEIGHT_MAX,
          int TILE_WIDTH_MAX,
          int AIE_VECTORIZATION_FACTOR,
          int CORES>
class xfcvDataMovers<KIND, DATA_TYPE, TILE_HEIGHT_MAX, TILE_WIDTH_MAX, AIE_VECTORIZATION_FACTOR, CORES, 0, true> {
    // using DataCopyF_t = std::function<int(DATA_TYPE*, DATA_TYPE*,
    // std::vector<int>&, int, int)>;

   private:
    uint16_t mOverlapH;
    uint16_t mOverlapV;
    uint16_t mTileRows;
    uint16_t mTileCols;
    bool mbUserHndl;

    cv::Mat* mpImage;
    DATA_TYPE* mpImgData;
    std::array<uint16_t, 3> mImageSize; // Rows, Cols, Elem Size

    std::vector<smartTileMetaData> mMetaDataList;

    xrtBufferHandle mImageBOHndl;

    //    DataCopyF_t mTileDataCopy;

    int imgSize() { return (mImageSize[0] * mImageSize[1] * mImageSize[2]); }

    int tileWindowSize() {
        return ((xf::cv::aie::METADATA_SIZE + (TILE_HEIGHT_MAX * TILE_WIDTH_MAX * sizeof(DATA_TYPE))));
    }

    int tileImgSize() { return (tileWindowSize() * (mTileRows * mTileCols)); }

    int bufferSizePerCore() { return (tileWindowSize() * ((mTileRows * mTileCols) / CORES)); }

    // Helper function for Tiler copy {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void input_copy(uint16_t startInd, uint16_t endInd) {
        assert(mpImgData);

        char* buffer = (char*)xrtBOMap(mImageBOHndl);
        int tileSize = tileWindowSize();
        for (int t = startInd; t < endInd; t++) {
            xf::cv::aie::metadata_elem_t* meta_data_p = (xf::cv::aie::metadata_elem_t*)(buffer + (t * tileSize));
            memset(meta_data_p, 0, xf::cv::aie::METADATA_SIZE);

            xf::cv::aie::xfSetTileWidth(meta_data_p, mMetaDataList[t].tileWidth());
            xf::cv::aie::xfSetTileHeight(meta_data_p, mMetaDataList[t].tileHeight());
            xf::cv::aie::xfSetTilePosH(meta_data_p, mMetaDataList[t].positionH());
            xf::cv::aie::xfSetTilePosV(meta_data_p, mMetaDataList[t].positionV());
            xf::cv::aie::xfSetTileOVLP_HL(meta_data_p, mMetaDataList[t].overlapSizeH_left());
            xf::cv::aie::xfSetTileOVLP_HR(meta_data_p, mMetaDataList[t].overlapSizeH_right());
            xf::cv::aie::xfSetTileOVLP_VT(meta_data_p, mMetaDataList[t].overlapSizeV_top());
            xf::cv::aie::xfSetTileOVLP_VB(meta_data_p, mMetaDataList[t].overlapSizeV_bottom());

            DATA_TYPE* image_data_p = (DATA_TYPE*)xf::cv::aie::xfGetImgDataPtr(meta_data_p);
            int16_t tileHeight = mMetaDataList[t].tileHeight();
            int16_t tileWidth = mMetaDataList[t].tileWidth();
            int16_t positionV = mMetaDataList[t].positionV();
            int16_t positionH = mMetaDataList[t].positionH();
            for (int ti = 0; ti < tileHeight; ti++) {
                memcpy(image_data_p + (ti * tileWidth), mpImgData + (((positionV + ti) * mImageSize[1]) + positionH),
                       tileWidth * sizeof(DATA_TYPE));
            }
        }
    }
    // }

    // Tiler copy {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void copy() {
        assert(mpImgData);

        uint16_t numThreads = std::thread::hardware_concurrency();

        std::thread mCopyThreads[numThreads];

        uint16_t tilesPerThread = (mTileRows * mTileCols) / numThreads;
        for (int i = 0; i < numThreads; i++) {
            uint16_t startInd = i * tilesPerThread;
            uint16_t endInd = (i == numThreads - 1) ? (mTileRows * mTileCols) : ((i + 1) * tilesPerThread);

            mCopyThreads[i] = std::thread(&xfcvDataMovers::input_copy, this, startInd, endInd);
        }
        for (int i = 0; i < numThreads; i++) {
            mCopyThreads[i].join();
        }
    }
    //}

    // Helper function for stitcher copy {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void output_copy(uint16_t startInd, uint16_t endInd) {
        assert(mpImgData != nullptr);

        char* buffer = (char*)xrtBOMap(mImageBOHndl);
        int tileSize = tileWindowSize();
        for (int t = startInd; t < endInd; t++) {
            xf::cv::aie::metadata_elem_t* meta_data_p = (xf::cv::aie::metadata_elem_t*)(buffer + (t * tileSize));

            int16_t tileWidth = xf::cv::aie::xfGetTileWidth(meta_data_p);
            int16_t tileHeight = xf::cv::aie::xfGetTileHeight(meta_data_p);
            int16_t positionH = xf::cv::aie::xfGetTilePosH(meta_data_p);
            int16_t positionV = xf::cv::aie::xfGetTilePosV(meta_data_p);
            int16_t overlapSizeH_left = xf::cv::aie::xfGetTileOVLP_HL(meta_data_p);
            int16_t overlapSizeH_right = xf::cv::aie::xfGetTileOVLP_HR(meta_data_p);
            int16_t overlapSizeV_top = xf::cv::aie::xfGetTileOVLP_VT(meta_data_p);
            int16_t overlapSizeV_bottom = xf::cv::aie::xfGetTileOVLP_VB(meta_data_p);

            int16_t correctedPositionH = positionH + overlapSizeH_left;
            int16_t correctedPositionV = positionV + overlapSizeV_top;
            int16_t correctedTileWidth =  tileWidth - (overlapSizeH_left + overlapSizeH_right);
            int16_t correctedTileHeight = tileHeight - (overlapSizeV_top + overlapSizeV_bottom);

            DATA_TYPE* image_data_p = (DATA_TYPE*)xf::cv::aie::xfGetImgDataPtr(meta_data_p);
            for (int ti = 0; ti < correctedTileHeight; ti++) {
                memcpy(mpImgData + (((correctedPositionV + ti) * mImageSize[1]) + correctedPositionH),
                       image_data_p + (((overlapSizeV_top + ti) * tileWidth) + overlapSizeH_left),
                       correctedTileWidth * sizeof(DATA_TYPE));
            }
        }
    }
    //}

    // Stitcher copy {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void copy() {
        assert(mpImgData != nullptr);

        uint16_t numThreads = std::thread::hardware_concurrency();

        std::thread mCopyThreads[numThreads];

        uint16_t tilesPerThread = (mTileRows * mTileCols) / numThreads;
        for (int i = 0; i < numThreads; i++) {
            uint16_t startInd = i * tilesPerThread;
            uint16_t endInd = (i == numThreads - 1) ? (mTileRows * mTileCols) : ((i + 1) * tilesPerThread);
            mCopyThreads[i] = std::thread(&xfcvDataMovers::output_copy, this, startInd, endInd);
        }
        for (int i = 0; i < numThreads; i++) {
            mCopyThreads[i].join();
        }
    }
    //}

    void free_buffer() {
        if (mImageBOHndl != nullptr) {
            xrtBOFree(mImageBOHndl);
        }
        mImageBOHndl = nullptr;
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void alloc_buffer() {
        if (mImageBOHndl == nullptr) {
            assert(tileImgSize() > 0);
            std::cout << "Allocating image device buffer (Tiler), "
                      << " Size : " << tileImgSize() << " bytes" << std::endl;
            //    mImageBOHndl = xrtBOAlloc(gpDhdl, tileImgSize(), XRT_BO_FLAGS_CACHEABLE, 0);
            mImageBOHndl = xrtBOAlloc(gpDhdl, tileImgSize(), 0, 0);
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void alloc_buffer() {
        if (mImageBOHndl == nullptr) {
            assert(tileImgSize() > 0);
            std::cout << "Allocating image device buffer (Stitcher), "
                      << " Size : " << tileImgSize() << " bytes" << std::endl;
            mImageBOHndl = xrtBOAlloc(gpDhdl, tileImgSize(), XRT_BO_FLAGS_CACHEABLE, 0);
        }
    }

    void regTilerStitcherCount() {
        if (KIND == TILER)
            ++gnTilerInstCount;
        else
            ++gnStitcherInstCount;
    }

   public:
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void start(std::array<std::string, CORES> portNames) {
        for (int i = 0; i < CORES; i++) {
            xrtBOSync(mImageBOHndl + i * bufferSizePerCore(), XCL_BO_SYNC_BO_TO_DEVICE, bufferSizePerCore(), 0);
            xrtSyncBOAIENB(gpDhdl, mImageBOHndl + i * bufferSizePerCore(), portNames[i].c_str(),
                           XCL_BO_SYNC_BO_GMIO_TO_AIE, bufferSizePerCore(), 0);
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    void wait(std::array<std::string, CORES> portNames) {
        for (int i = 0; i < CORES; i++) {
            xrtGMIOWait(gpDhdl, portNames[i].c_str());
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void start(std::array<std::string, CORES> portNames) {
        for (int i = 0; i < CORES; i++) {
            xrtSyncBOAIENB(gpDhdl, mImageBOHndl + i * bufferSizePerCore(), portNames[i].c_str(),
                           XCL_BO_SYNC_BO_AIE_TO_GMIO, bufferSizePerCore(), 0);
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void wait(std::array<std::string, CORES> portNames) {
        for (int i = 0; i < CORES; i++) {
            xrtGMIOWait(gpDhdl, portNames[i].c_str());
            xrtBOSync(mImageBOHndl + i * bufferSizePerCore(), XCL_BO_SYNC_BO_FROM_DEVICE, bufferSizePerCore(), 0);
        }

        // Copy data from device buffer to host
        copy();

        CtypeToCVMatType<DATA_TYPE> type;
        if (mpImage != nullptr) {
            cv::Mat dst(mImageSize[0], mImageSize[1], type.type, mpImgData);

            // TODO: saturation to be done based on the mat type ???
            if (mpImage->type() == CV_8U) {
                // Saturate the output values to [0,255]
                dst = cv::max(dst, 0);
                dst = cv::min(dst, 255);
            }
            dst.convertTo(*mpImage, mpImage->type());
        }
        mpImage = nullptr;
    }

    // Initialization / device buffer allocation / tile header copy / type
    // conversion to be done in constructor {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    xfcvDataMovers(uint16_t overlapH, uint16_t overlapV) {
        if (gpDhdl == nullptr) {
            throw std::runtime_error("No valid device handle found. Make sure using xF::deviceInit(...) is called.");
        }

        mpImgData = nullptr;
        mImageSize = {0, 0, 0};

        // Initialize overlaps
        mOverlapH = overlapH;
        mOverlapV = overlapV;

        mTileRows = 0;
        mTileCols = 0;
        mbUserHndl = false;

        mImageBOHndl = nullptr;

        // Register the count of tiler/stitcher objects
        regTilerStitcherCount();
    }
    //}

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    xfcvDataMovers() {
        if (gpDhdl == nullptr) {
            throw std::runtime_error("No valid device handle found. Make sure using xF::deviceInit(...) is called.");
        }

        mpImage = nullptr;
        mpImgData = nullptr;
        mImageSize = {0, 0, 0};

        // Initialize overlaps
        mOverlapH = 0;
        mOverlapV = 0;

        mTileRows = 0;
        mTileCols = 0;
        mbUserHndl = false;

        mImageBOHndl = nullptr;

        // Register the count of tiler/stitcher objects
        regTilerStitcherCount();
    }

    // Non copyable {
    xfcvDataMovers(const xfcvDataMovers&) = delete;
    xfcvDataMovers& operator=(const xfcvDataMovers&) = delete;
    //}

    //    void setTileCopyFn(DataCopyF_t& fn);

    // Close / free operations tp be done here {
    ~xfcvDataMovers() {
        free_buffer();
        if (gpDhdl != nullptr) {
            xrtDeviceClose(gpDhdl);
        }
        gpDhdl = nullptr;
    }
    //}

    void compute_metadata(const cv::Size& inImgSize, const cv::Size& outImgSize = cv::Size(0,0));

    // These functions will start the data transfer protocol {
    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    std::array<uint16_t, 2> host2aie_nb(DATA_TYPE* img_data,
                                        const cv::Size& img_size,
                                        std::array<std::string, CORES> portNames) {
        int old_img_buffer_size = imgSize();

        bool bRecompute = false;
        if ((mImageSize[0] != img_size.height) || (mImageSize[1] != img_size.width)) {
            bRecompute = true;
        }

        mpImgData = (DATA_TYPE*)img_data;
        mImageSize = {(uint16_t)img_size.height, (uint16_t)img_size.width, (uint16_t)sizeof(DATA_TYPE)};

        if (bRecompute == true) {
            // Pack metadata
            compute_metadata(img_size);
        }

        int new_img_buffer_size = imgSize();

        if ((new_img_buffer_size > old_img_buffer_size)) {
            free_buffer();
        }

        // Allocate buffer
        alloc_buffer();

        // Copy input data to device buffer
        copy();

        // Start the data transfers
        start(portNames);

        std::array<uint16_t, 2> ret = {mTileRows, mTileCols};

        return ret;
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    std::array<uint16_t, 2> host2aie_nb(cv::Mat& img, std::array<std::string, CORES> portNames) {
        CtypeToCVMatType<DATA_TYPE> cType;

        if (cType.type == img.type()) {
            return host2aie_nb((DATA_TYPE*)img.data, img.size(), portNames);
        } else if (cType.type < img.type()) {
            cv::Mat temp;
            img.convertTo(temp, cType.type);
            return host2aie_nb((DATA_TYPE*)temp.data, img.size(), portNames);
        } else {
            std::vector<DATA_TYPE> imgData;
            imgData.assign(img.data, img.data + img.total());
            return host2aie_nb(imgData.data(), img.size(), portNames);
        }
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void aie2host_nb(DATA_TYPE* img_data,
                     const cv::Size& img_size,
                     std::array<uint16_t, 2> tiles,
                     std::array<std::string, CORES> portNames) {
        int old_img_buffer_size = imgSize();

        mpImgData = (DATA_TYPE*)img_data;
        mImageSize = {(uint16_t)img_size.height, (uint16_t)img_size.width, sizeof(DATA_TYPE)};

        mTileRows = tiles[0];
        mTileCols = tiles[1];

        int new_img_buffer_size = imgSize();
        if ((new_img_buffer_size > old_img_buffer_size)) {
            free_buffer();
        }

        // Allocate buffer
        alloc_buffer();

        // Start the kernel
        start(portNames);
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void aie2host_nb(cv::Mat& img, std::array<uint16_t, 2> tiles, std::array<std::string, CORES> portNames) {
        mpImage = &img;

        CtypeToCVMatType<DATA_TYPE> cType;

        if (cType.type == img.type()) {
            return aie2host_nb((DATA_TYPE*)img.data, img.size(), tiles, portNames);
        }

        DATA_TYPE* imgData = (DATA_TYPE*)malloc(img.size().height * img.size().width * sizeof(DATA_TYPE));

        aie2host_nb(imgData, img.size(), tiles, portNames);
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    std::array<uint16_t, 2> host2aie(cv::Mat& img, std::array<std::string, CORES> portNames) {
        std::array<uint16_t, 2> ret = host2aie_nb(img, portNames);

        wait(portNames);

        return ret;
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
    std::array<uint16_t, 2> host2aie(DATA_TYPE* img_data,
                                     const cv::Size& img_size,
                                     std::array<std::string, CORES> portNames) {
        std::array<uint16_t, 2> ret = host2aie_nb(img_data, img_size, portNames);

        wait(portNames);

        return ret;
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void aie2host(cv::Mat& img, std::array<uint16_t, 2> tiles, std::array<std::string, CORES> portNames) {
        aie2host_nb(img, tiles, portNames);

        wait(portNames);
    }

    template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
    void aie2host(DATA_TYPE* img_data,
                  const cv::Size& img_size,
                  std::array<uint16_t, 2> tiles,
                  std::array<std::string, CORES> portNames) {
        aie2host_nb(img_data, img_size, tiles, portNames);

        wait(portNames);
    }

    //}
};

/*
template <DataMoverKind KIND,
          typename DATA_TYPE,
          int TILE_HEIGHT_MAX,
          int TILE_WIDTH_MAX,
          int AIE_VECTORIZATION_FACTOR,
          int CORES>
void xfcvDataMovers<KIND, DATA_TYPE, TILE_HEIGHT_MAX, TILE_WIDTH_MAX,
AIE_VECTORIZATION_FACTOR, CORES, 0, true>::
    setTileCopyFn(DataCopyF_t& fn) {}
*/
template <DataMoverKind KIND,
          typename DATA_TYPE,
          int TILE_HEIGHT_MAX,
          int TILE_WIDTH_MAX,
          int AIE_VECTORIZATION_FACTOR,
          int CORES>
void xfcvDataMovers<KIND, DATA_TYPE, TILE_HEIGHT_MAX, TILE_WIDTH_MAX, AIE_VECTORIZATION_FACTOR, CORES, 0, true>::
    compute_metadata(const cv::Size& inImgSize, const cv::Size& outImgSize) {
    mMetaDataList.clear();
    
    cv::Size inputImgSize = inImgSize;
    cv::Size outputImgSize = outImgSize;

    // Set default output image size to input image size
    if (outputImgSize == cv::Size(0,0)) {
      outputImgSize = inputImgSize;
    }

    mImageSize[0] = (uint16_t)inputImgSize.height;
    mImageSize[1] = (uint16_t)inputImgSize.width;

    if (inputImgSize.height == outputImgSize.height && inputImgSize.width == outputImgSize.width) {
        smartTileTilerGenerateMetaDataWithSpecifiedTileSize(
            {inputImgSize.height, inputImgSize.width}, mMetaDataList, mTileRows, mTileCols, {TILE_HEIGHT_MAX, TILE_WIDTH_MAX},
            {mOverlapH, mOverlapH}, {mOverlapV, mOverlapV}, AIE_VECTORIZATION_FACTOR, true);
    } else {
        smartTileTilerGenerateMetaDataWithSpecifiedTileSize(
            {inputImgSize.height, inputImgSize.width}, {outputImgSize.height,outputImgSize.width}, mMetaDataList, mTileRows, mTileCols,
            AIE_VECTORIZATION_FACTOR, true);
    }
    char sMesg[2048];
    sMesg[0] = '\0';
    sprintf(sMesg, "Requested tile size (%d,%d). Computed tile size (%d,%d). Number of tiles (%d,%d)\n",
            TILE_HEIGHT_MAX, TILE_WIDTH_MAX, mMetaDataList[0].tileHeight(), mMetaDataList[0].tileWidth(), mTileRows,
            mTileCols);
    std::cout << sMesg << std::endl;
}

} // xF

#endif

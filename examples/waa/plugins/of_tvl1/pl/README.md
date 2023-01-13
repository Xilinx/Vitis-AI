# TVL1 Optical Flow

TVL1 is used for Optical flow application and its formulation is based on Total Variation (TV) regularization & the robust L1 normalization in the data fidelity term. TVL1 algorithm preserves discontinuties in the flow field and increases robustness against occlusions, illumination changes and noise.

## API description

### TVL1 class reference

* create()

	static Ptr<DualTVL1OpticalFlow> xf::cv::DualTVL1OpticalFlow::create    (    double     tau = 0.25, double     lambda = 0.15, double     theta = 0.3, int     nscales = 5, int     warps = 5, double     epsilon = 0.01, int     innnerIterations = 30, int     outerIterations = 10, double     scaleStep = 0.8,   int     medianFiltering = 5)

* calc()

    xf::cv::DualTVL1OpticalFlow::calc    ( frame1, flow ) //TVL1 processing 

* getEpsilon(), setEpsilon() 
  
  	Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.

* getInnerIterations() & setInnerIterations() 

	Inner iterations (between outlier filtering) used in the numerical scheme. Default value is used from create().

* getLambda() & setLambda() 

	Weight parameter for the data term, attachment parameter. Default value is used from create().

* getOuterIterations() & setOuterIterations() 

	Outer iterations (number of inner loops) used in the numerical scheme. Default value is used from create().

* getScalesNumber() & setScalesNumber() 

	Number of scales used to create the pyramid of images. Default value is used from create().

* getScaleStep() & setScaleStep() 

	Step between scales (<1). Default value is used from create().

* getTau() & setTau() 

	Time step of the numerical scheme. Default value is used from create().

* getTheta() & setTheta() 

	Weight parameter for (u - v)^2, tightness parameter. Default value is used from create().

* getWarpingsNumber() & setWarpingsNumber()

	Number of warpings per scale. Default value is used from create().

## OpenCV Installation Guidance:

It is recommended to do a fresh installation of OpenCV 4.4.0 and not use existing libs of your system, as they may or may not work with Vitis environment.

**Please make sure you update and upgrade the packages and OS libraries of your system and
have cmake version>3.5 installed before proceeding.**

The below steps can help install the basic libs required to compile and link the OpenCV calls in Vitis Vision host code.

1. create a directory "source" and clone [opencv-4.4.0](https://github.com/opencv/opencv/tree/4.4.0) into it.
2. create a directory "source_contrib" and clone [opencv-4.4.0-contrib](https://github.com/opencv/opencv_contrib/tree/4.4.0) into it.
3. create 2 more directories: *build* , *install*
4. open a bash terminal and *cd* to *build* directory
5. Run the command: *export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/*
6. Run the command: *cmake -D CMAKE_BUILD_TYPE=RELEASE
  -D CMAKE_INSTALL_PREFIX=< path-to-install-directory>
  -D CMAKE_CXX_COMPILER=< path-to-Vitis-installation-directory>/tps/lnx64/gcc-6.2.0/bin/g++
  -D OPENCV_EXTRA_MODULES_PATH=< path-to-source_contrib-directory>/modules/
  -D WITH_V4L=ON -DBUILD_TESTS=OFF -DBUILD_ZLIB=ON
  -DBUILD_JPEG=ON -DWITH_JPEG=ON -DWITH_PNG=ON
  -DBUILD_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF
  -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_OPENEXR=OFF
  -DBUILD_OPENEXR=OFF <path-to-source-directory>*
7. Run the command: *make all -j8*
8. Run the command: *make install*

The OpenCV includes and libs will be in the *install* directory

## Step to create U50 xclbin & build the application

```
source < path-to-Vitis-installation-directory >/Vitis/2021.1/settings64.sh
source < path-to-XRT-installation-directory >/setup.sh
export OPENCV_INCLUDE=< path-to-opencv-include-folder >/include
export OPENCV_LIB=< path-to-opencv-lib-folder >/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >/lib64
export DEVICE=< path-to-platform-directory >/xilinx_u50_gen3x4_xdma_2_202010_1/xilinx_u50_gen3x4_xdma_2_202010_1.xpfm
make clean

#To generate xclbin
make xclbin TARGET=hw

#To build the application
make host TARGET=hw
```
Note that 
- Generated xclbin will be here **${VAI_HOME}/examples/Whole-App-Acceleration/plugins/of_tvl1/pl/krnl_tvl1.xclbin**.
- Executable file will be here **${VAI_HOME}/examples/Whole-App-Acceleration/plugins/of_tvl1/pl/build_dir.hw.xilinx_u50_gen3x4_xdma_2_202010_1/tvl1**.
- Build time ~ 5 hrs


## Download image dataset
- Download image dataset using [CAVIAR Test Case Scenarios](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
	```
	wget http://groups.inf.ed.ac.uk/vision/CAVIAR/CAVIARDATA2/EnterExitCrossingPaths1cor/EnterExitCrossingPaths1cor.tar.gz --no-check-certificate
	mkdir -p input_img_dir
	tar -xzvf EnterExitCrossingPaths1cor.tar.gz -C input_img_dir/
	```

## Run TVL1 example
* Run with SW TVL1 optical flow
	```
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >
	mkdir -p output_img_dir

	./build_dir.hw.xilinx_u50_gen3x4_xdma_2_202010_1/tvl1 ./input_img_dir/ ./output_img_dir/ krnl_tvl1.xclbin 0

	Expect:
	Run with SW TVL1 optical flow
	Total frame: 383  Performance: 1.26697 fps
	```
* Run with TVL1 optical flow Hardware Accelerator
	```
	source /opt/xilinx/xrt/setup.sh
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >
	mkdir -p output_img_dir

	./build_dir.hw.xilinx_u50_gen3x4_xdma_2_202010_1/tvl1 ./input_img_dir/ ./output_img_dir/ krnl_tvl1.xclbin 1

	Expect:
	Run with HW accelerator:TVL1 optical flow
	Found Platform
	Platform Name: Xilinx
	INFO: Importing krnl_tvl1.xclbin
	Loading: 'krnl_tvl1.xclbin'
	Total frame: 383  Performance: 10.4594 fps
	```
## Performance
Below table shows the comparison of performance achieved by accelerating TVL1. Performance numbers are achieved by running 383 images of resolution 384x288.

<table style="undefined;table-layout: fixed; width: 534px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">FPGA</th>
    <th colspan="2">FPS</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement in throughput</span></th>
  </tr>
  <tr>
    <td>Software TVL1</td>
    <td>Hardware Accelerated TVL1</td>
  </tr>


  
  <tr>
   <td>Alveo-U50</td>
    <td>1.27</td>
    <td>10.46</td>
        <td>723.6%</td>
  </tr>

</table>

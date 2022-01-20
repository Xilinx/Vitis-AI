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



#pragma once

#include <iostream>
#include "common/xf_headers.hpp"
#include "xcl2.hpp"

namespace xf {
  namespace cv {

    class DualTVL1OpticalFlow {

        private :

            // TVL1 Algo related
            double Tau, Lambda, Theta, Epsilon, ScaleStep, Gamma;
            int    NScales, Warps, InnerIterations, OuterIterations, MedianFiltering;
            bool   UseInitialFlow;

            // OpenCL implementation related
            cl::Buffer       FrameBuffer[3];   // OpenCL Input Frame buffer
            cl::Buffer       PyramidParamBuff; // OpenCL Resize paramters buffer
            cl::Buffer       AlgoParamBuff;    // OpenCL TVL1 Algo paramters buffer
            cl::Buffer       FlowBuffer;       // OpenCL Output Flow buffer
            cl::Buffer       ErrBuffer;       // OpenCL Output Flow buffer
            cl::Buffer       TmpBuffer[10];    // OpenCL Temporary buffer(s)
            cl::CommandQueue queue;            // OpenCL Command Q
            cl::Kernel       PyramidKernel;    // Image Pyramid generation H/W Kernel
            cl::Kernel       TVL1AlgoKernel;   // TVL1 Algo execution H/W Kernel
            cl_int           err;              // For error handling

            //bool      StartAlgo;
            int       PyramidBuffIdx, AlgoBuffIdx, NxtAlgoBuffIdx;
            uint8_t  *ImgData[3];
            uint32_t *FlowData;
            uint32_t *ErrData;
	    int       FrameCount;

            // ---------------------------------------------------------------------------------------------
            // Constructor
            // ---------------------------------------------------------------------------------------------
            DualTVL1OpticalFlow () {
                Tau             = 0.25;
                Lambda          = 0.15;
                Theta           = 0.3;
                NScales         = 5;
                Warps           = 5;
                Epsilon         = 0.01;
                InnerIterations = 30;
                OuterIterations = 10;
                ScaleStep       = 0.8;
                Gamma           = 0.0;
                MedianFiltering = 5;
                UseInitialFlow  = false;

                // Internal variables
                //StartAlgo      = false;
                PyramidBuffIdx = 0;
                AlgoBuffIdx    = 1;
                NxtAlgoBuffIdx = 2;
		FrameCount = 0;
            }


            // ---------------------------------------------------------------------------------------------
            // Compute next buffer index
            // ---------------------------------------------------------------------------------------------
            int nextBufferIdx (int Idx) {
                return (Idx==2 ? 0 : Idx==1 ? 2 : 1); // = (Idx+1) % 3
            }

            // ---------------------------------------------------------------------------------------------
            // Conversion to Float -> Fixed (Q(32-FBITS).FBITS) format
            // ---------------------------------------------------------------------------------------------
	    template<int FBITS=16>
            uint32_t toFixed(float Val) {
                return (uint32_t)(Val * (1 << FBITS));
            }

            // ---------------------------------------------------------------------------------------------
            // Function to wait for Command Q to be empty
            // ---------------------------------------------------------------------------------------------
            void waitForQueueEmpty(std::string extra_msg = "") {

              if (extra_msg != "") {
                extra_msg = "("+extra_msg+") ";
              }

              queue.finish();
            }

        public :
		
            ~DualTVL1OpticalFlow () {
                release();
            }


            // ---------------------------------------------------------------------------------------------
            // Function to initilize Xilinx device
            // ---------------------------------------------------------------------------------------------
            //static void init (std::string& XCLBinaryName, int _height, int _width) {
            void init (std::string& XCLBinaryName, int _height, int _width) {
                // Initial setup
                // .........................................................

		FrameCount = 0;

                // Get the device details:
                // -------------------------
                // Device ID
                std::vector<cl::Device> devices = xcl::get_xil_devices();
                cl::Device device = devices[0];

                // Device Name
                OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

                // Create Context:
                // -------------------------
                OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));

                // Load binary file:
                // -------------------------
                cl::Program::Binaries bins = xcl::import_binary_file(XCLBinaryName);

                // Create a program:
                // -------------------------
                devices.resize(1); // Removes all other devices
                OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

                // Create Command Queue:
                // -------------------------
                OCL_CHECK(err, cl::CommandQueue _queue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
                queue = _queue;

                // Create a kernel(s):
                // -------------------------
                OCL_CHECK(err, cl::Kernel _krnl0(program, "img_pyramid_accel", &err));
                PyramidKernel = _krnl0;

                OCL_CHECK(err, cl::Kernel _krnl1(program, "tvl1_accel", &err));
                TVL1AlgoKernel = _krnl1;

                // .........................................................
                // Pre-processing:
                //
                // + Creating CL buffers for (device memory allocation)
                //    - Parameters
                //    - Pyramid Buffers
                //
                // + Setting required parameters
                // .........................................................

                // Calculating buffer sizes
                cl::size_type    _ParamSize = (cl::size_type)(((4 * NScales) + 5) * sizeof(uint32_t)); // No.of Levels + ((row + col + offset) x (No.of levels))
                cl::size_type _AlgoParamSize = (cl::size_type)(                  16 * sizeof(uint32_t)); // A total of 16 elements

                // CL Buffer creation (device memory allocation)
                OCL_CHECK(err, cl::Buffer     _ParamBuff(context, CL_MEM_READ_WRITE,     _ParamSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer _AlgoParamBuff(context, CL_MEM_READ_WRITE, _AlgoParamSize, NULL, &err));

                // Saving for future
                PyramidParamBuff = _ParamBuff;
                AlgoParamBuff    = _AlgoParamBuff;

                // Calculating scaled image resolutions and copying into parameter list
                // .....................................................................
                float    scale = 1.0f; // Pyramid's 1st level scalefactor
                uint32_t level_offset = 0;

                uint32_t* ParamData = (uint32_t*) queue.enqueueMapBuffer(PyramidParamBuff, CL_TRUE, CL_MAP_WRITE, 0, _ParamSize);

	        int c_NUM_WORDS_FLOW_PTR = 8;//NPC_TVL_INNERCORE;
	        int c_NUM_BYTES_IMG_PTR  = IMAGE_PTR_WIDTH_P/8;
                int height = _height;
                int width  = _width ;
		int level;
                for (level = 0; level < NScales+1; level++) {
		   int width_align;
		   if(level==0)
		      width_align = width;
		   else
		      width_align = ((width + c_NUM_WORDS_FLOW_PTR - 1) / c_NUM_WORDS_FLOW_PTR) * c_NUM_WORDS_FLOW_PTR;
                      
		   // Parameters (row x col) of level-i image
                   ParamData[4*level+1] = (int)height;
                   ParamData[4*level+2] = (int)width_align;
                   ParamData[4*level+3] = (int)level_offset;
		   float err_threshold = Epsilon*Epsilon*height*width_align;
		   float err_th_fxd = round(err_threshold*(1<<(ERROR_BW/2)));
                   ParamData[4*level+4] = (int)err_th_fxd;

                   level_offset += ((height * width_align + c_NUM_BYTES_IMG_PTR - 1)/c_NUM_BYTES_IMG_PTR);

                   // Prepare image dimensions for next level
		   if(level!=0)
		   {
                     height = height*ScaleStep;
                     width  = width*ScaleStep;
		   }  

		   if((height<16) || (width<16))
		      break;
                }
                ParamData[0] = level - 1; // Total no.of levels
		level_offset *= c_NUM_BYTES_IMG_PTR;

		//for(int i=0;i<1+4*(NScales+1);i++)
		//{
		//   std::cout << "param[" << i << "] " << (int)ParamData[i] << std::endl;
		//}
		//std::cout << "level_offset  " << level_offset << std::endl;

                queue.enqueueUnmapMemObject(PyramidParamBuff, ParamData);

                // TVL1 Algo parameters
                // .....................................................................
                uint32_t* AlgoParamData = (uint32_t*) queue.enqueueMapBuffer(AlgoParamBuff, CL_TRUE, CL_MAP_WRITE, 0, _AlgoParamSize);

                AlgoParamData[0]  = toFixed<FLOW_F_BITS>(Lambda*Theta);
                AlgoParamData[1]  = toFixed<FLOW_F_BITS>(Theta);
                AlgoParamData[2]  = toFixed<FLOW_F_BITS>(Gamma);
                AlgoParamData[3]  = toFixed<FLOW_F_BITS>(Epsilon);
                AlgoParamData[4]  = toFixed<FLOW_F_BITS>(Tau/Theta);
                AlgoParamData[5]  = toFixed<FLOW_F_BITS>(1.0f / ScaleStep);
                AlgoParamData[6]  = InnerIterations;
                AlgoParamData[7]  = OuterIterations;
                AlgoParamData[8]  = Warps;
                AlgoParamData[9]  = (_height*_width + NPC_TVL_INNERCORE - 1 )/NPC_TVL_INNERCORE;
                AlgoParamData[10]  = 0;//Error Debug Enable

                queue.enqueueUnmapMemObject(AlgoParamBuff, AlgoParamData);

                // Creating references to CL buffer
                // .....................................................................

                cl::size_type         _ImgSize = (cl::size_type)(_height * _width * sizeof( uint8_t));
                cl::size_type         _ErrSize = (cl::size_type)(10000 * sizeof( int));
                cl::size_type        _FlowSize = (cl::size_type)(_height * _width * 2 * sizeof(uint32_t));
                cl::size_type _FrameBufferSize = (cl::size_type)(level_offset     * sizeof( uint8_t));
                cl::size_type   _TmpBufferSize = (cl::size_type)(_height * _width * sizeof(uint32_t)); 

                OCL_CHECK(err, cl::Buffer _FrameBuffer0(context, CL_MEM_READ_WRITE, _FrameBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer _FrameBuffer1(context, CL_MEM_READ_WRITE, _FrameBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer _FrameBuffer2(context, CL_MEM_READ_WRITE, _FrameBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer   _FlowBuffer(context, CL_MEM_READ_WRITE,        _FlowSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer   _ErrBuffer(context, CL_MEM_READ_WRITE,         _ErrSize, NULL, &err));

                OCL_CHECK(err, cl::Buffer     _U1Buffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize*2, NULL, &err));
                OCL_CHECK(err, cl::Buffer     _U2Buffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize*2, NULL, &err));
                OCL_CHECK(err, cl::Buffer   _I1wxBuffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer   _I1wyBuffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer   _gradBuffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer   _rhocBuffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer    _p11Buffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer    _p12Buffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer    _p21Buffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));
                OCL_CHECK(err, cl::Buffer    _p22Buffer(context, CL_MEM_READ_WRITE,   _TmpBufferSize, NULL, &err));

                FrameBuffer[0] = _FrameBuffer0;
                FrameBuffer[1] = _FrameBuffer1;
                FrameBuffer[2] = _FrameBuffer2;
                FlowBuffer     = _FlowBuffer;
                ErrBuffer     = _ErrBuffer;

                // Temp Buffers
                TmpBuffer[0] = _U1Buffer;
                TmpBuffer[1] = _U2Buffer;
                TmpBuffer[2] = _I1wxBuffer;
                TmpBuffer[3] = _I1wyBuffer;
                TmpBuffer[4] = _gradBuffer;
                TmpBuffer[5] = _rhocBuffer;
                TmpBuffer[6] = _p11Buffer;
                TmpBuffer[7] = _p12Buffer;
                TmpBuffer[8] = _p21Buffer;
                TmpBuffer[9] = _p22Buffer;

                ImgData[0] = (uint8_t  *)queue.enqueueMapBuffer(FrameBuffer[0], CL_TRUE, CL_MAP_WRITE, 0, _ImgSize);
                ImgData[1] = (uint8_t  *)queue.enqueueMapBuffer(FrameBuffer[1], CL_TRUE, CL_MAP_WRITE, 0, _ImgSize);
                ImgData[2] = (uint8_t  *)queue.enqueueMapBuffer(FrameBuffer[2], CL_TRUE, CL_MAP_WRITE, 0, _ImgSize);
                FlowData   = (uint32_t *)queue.enqueueMapBuffer(FlowBuffer,     CL_TRUE, CL_MAP_READ,  0, _FlowSize);
                ErrData   = (uint32_t *)queue.enqueueMapBuffer(ErrBuffer,     CL_TRUE, CL_MAP_READ,    0, _ErrSize);

                waitForQueueEmpty("pre-process init");

                OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({PyramidParamBuff, AlgoParamBuff}, 0)); // 0 means from host

                waitForQueueEmpty("pre-process ParamBuff Migrate");

                // .........................................................
                // Setting one-time kernel(s) arguments
                // .........................................................
                OCL_CHECK(err, err =  PyramidKernel.setArg( 2, PyramidParamBuff));

                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 2, FlowBuffer));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 3, TmpBuffer[0]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 4, TmpBuffer[1]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 5, TmpBuffer[2]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 6, TmpBuffer[3]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 7, TmpBuffer[4]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 8, TmpBuffer[5]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg( 9, TmpBuffer[6]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg(10, TmpBuffer[7]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg(11, TmpBuffer[8]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg(12, TmpBuffer[9]));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg(13, AlgoParamBuff));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg(14, PyramidParamBuff));
                OCL_CHECK(err, err = TVL1AlgoKernel.setArg(15, ErrBuffer));

                return;
            }

	    void resetFrameCount() {
		FrameCount = 0;
	    }

            // ---------------------------------------------------------------------------------------------
            // Function to clean-up at exit
            // ---------------------------------------------------------------------------------------------
            void release() {
              // release CL buffers
              queue.enqueueUnmapMemObject(FrameBuffer[0], ImgData[0]);
              queue.enqueueUnmapMemObject(FrameBuffer[1], ImgData[1]);
              queue.enqueueUnmapMemObject(FrameBuffer[2], ImgData[2]);
              queue.enqueueUnmapMemObject(    FlowBuffer,   FlowData);
              queue.enqueueUnmapMemObject(     ErrBuffer,    ErrData);

              waitForQueueEmpty("release");

              return;
            }

            // Methods for accessing class members
            double getTau ()                    { return Tau;                 }
            double getLambda ()                 { return Lambda;              }
            double getTheta ()                  { return Theta;               }
            int    getScalesNumber ()           { return NScales;             }
            int    getWarpingsNumber ()         { return Warps;               }
            double getEpsilon ()                { return Epsilon;             }
            int    getInnerIterations ()        { return InnerIterations;     }
            int    getOuterIterations ()        { return OuterIterations;     }
            double getScaleStep ()              { return ScaleStep;           }
            double getGamma ()                  { return Gamma;               }
            int    getMedianFiltering ()        { return MedianFiltering;     }
            bool   getUseInitialFlow ()         { return UseInitialFlow;      }

	    void   setTau (double val)          { this->Tau = val;             }
            void   setLambda (double val)       { this->Lambda = val;          }
            void   setTheta (double val)        { this->Theta = val;           }
            void   setScalesNumber (int val)    { this->NScales = val;         }
            void   setWarpingsNumber (int val)  { this->Warps = val;           }
            void   setEpsilon (double val)      { this->Epsilon = val;         }
            void   setInnerIterations (int val) { this->InnerIterations = val; }
            void   setOuterIterations (int val) { this->OuterIterations = val; }
            void   setScaleStep (double val)    { this->ScaleStep = val;       }
            void   setGamma (double val)        { this->Gamma = val;           }
            void   setMedianFiltering (int val) { this->MedianFiltering = val; }
            void   setUseInitialFlow (bool val) { this->UseInitialFlow = val;  }

            // Creates an instance of DualTVL1OpticalFlow and sets its parameters
            static ::cv::Ptr<DualTVL1OpticalFlow> create (
                double tau=0.25, double lambda=0.15, double theta=0.3, int nscales=5, int warps=5,
                double epsilon=0.01, int innnerIterations=30, int outerIterations=10, double scaleStep=0.8,
                double gamma=0.0, int medianFiltering=5, bool useInitialFlow=false
            ) {
                ::cv::Ptr<DualTVL1OpticalFlow> of = new DualTVL1OpticalFlow;

                of->setTau             (tau             );
                of->setLambda          (lambda          );
                of->setTheta           (theta           );
                of->setScalesNumber    (nscales         );
                of->setWarpingsNumber  (warps           );
                of->setEpsilon         (epsilon         );
                of->setInnerIterations (innnerIterations);
                of->setOuterIterations (outerIterations );
                of->setScaleStep       (scaleStep       );
                of->setGamma           (gamma           );
                of->setMedianFiltering (medianFiltering );
                of->setUseInitialFlow  (useInitialFlow  );

                return of;
            }

            // Calculates Optical flow using TVL1 algorithm
            void calc(::cv::InputArray I0, ::cv::InputArray I1, ::cv::InputOutputArray flow);

            //  - Optimized verison for Xilinx device
            void calc(::cv::Mat Frame, ::cv::Mat &Flow);
    };

    void DualTVL1OpticalFlow::calc(::cv::Mat Frame, ::cv::Mat &Flow) {

	FrameCount++;

	Flow.create(Frame.rows, Frame.cols, Flow.type());

        // For event base profiling...
        cl::Event PyramidKernelEvent, AlgoKernelEvent;

        // Copying in image to device memory
        ::cv::Mat Image(Frame.size(), Frame.type(), ImgData[PyramidBuffIdx]);
        Frame.copyTo(Image);

        OCL_CHECK(err, err =  PyramidKernel.setArg(0, FrameBuffer[PyramidBuffIdx]));
        OCL_CHECK(err, err =  PyramidKernel.setArg(1, FrameBuffer[PyramidBuffIdx]));

        OCL_CHECK(err, err = TVL1AlgoKernel.setArg(0, FrameBuffer[   AlgoBuffIdx]));
        OCL_CHECK(err, err = TVL1AlgoKernel.setArg(1, FrameBuffer[NxtAlgoBuffIdx]));

        // Migrate input data to kernel space
        OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({FrameBuffer[PyramidBuffIdx]}, 0)); // 0 means from host

        waitForQueueEmpty("input MigrateMemObjects");

        // Lanch kernel(s)
        OCL_CHECK(err, err = queue.enqueueTask(PyramidKernel,  NULL, &PyramidKernelEvent));
	if(FrameCount>2)
          OCL_CHECK(err, err = queue.enqueueTask(TVL1AlgoKernel, NULL, &AlgoKernelEvent));

        #if __XF_PROFILE_KERNEL__
        // Wait for the kernel event. This indicates kernel is done
        clWaitForEvents(1, (const cl_event*)&PyramidKernelEvent);
        clWaitForEvents(1, (const cl_event*)&AlgoKernelEvent);


        cl_ulong start = 0;
        cl_ulong end   = 0;

        PyramidKernelEvent.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        PyramidKernelEvent.getProfilingInfo(CL_PROFILING_COMMAND_END,     &end);

        std::cout << "[Info] Actual Pyramid kernel latency (ms): " << ((end - start) / 1000000.0) << std::endl;

        start = 0;
        end   = 0;

        AlgoKernelEvent.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        AlgoKernelEvent.getProfilingInfo(CL_PROFILING_COMMAND_END,     &end);

        std::cout << "[Info] Actual TVL1 Algo kernel latency (ms): " << ((end - start) / 1000000.0) << std::endl;
        #endif

        waitForQueueEmpty("kernel(s) execution");

        // Moved the output data back to host
	if(FrameCount>2)
	{
           OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({FlowBuffer}, CL_MIGRATE_MEM_OBJECT_HOST));
           OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({ErrBuffer}, CL_MIGRATE_MEM_OBJECT_HOST));
           waitForQueueEmpty("data move to host");
	}

        // Copy output
        ::cv::Mat FlowOut(Flow.size(), Flow.type(), FlowData);
        FlowOut.copyTo(Flow);

        PyramidBuffIdx = nextBufferIdx(PyramidBuffIdx);
        AlgoBuffIdx    = nextBufferIdx(AlgoBuffIdx);
        NxtAlgoBuffIdx = nextBufferIdx(NxtAlgoBuffIdx);


    } // end of calc

  } // end of namespace cv
} // end of namesapce xf

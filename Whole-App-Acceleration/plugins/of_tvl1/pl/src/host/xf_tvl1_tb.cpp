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

#include "tvl1_header.hpp"

int main(int argc, char** argv) {

    	if (argc != 5) {
          fprintf(stderr, "Usage: %s <Input_image_directory> <Output_image_directory> <XCLBIN_File> <1/0 to enable HW Aceelerator> \n", argv[0]);
          return EXIT_FAILURE;
        }

        std::string in_img_dir   = argv[1];
        std::string out_flow_dir = argv[2];
        std::string xclbin_path  = argv[3];
        bool enable_hw_accelerator = stoi(argv[4]);

        vector<cv::String> fn;
        string jpg_img = "/*.jpg";
        glob(in_img_dir+jpg_img, fn, false);

        long calc_time = 0;
        int calc_cnt = 0;

	if(enable_hw_accelerator==0)
	{
           std::cout << "Run with SW TVL1 optical flow" << std::endl;
	   Mat_<Point2f> flow;
	   Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::DualTVL1OpticalFlow::create();
	
           for (int img_cnt=0;img_cnt<fn.size()-1;img_cnt++) 
           {
              cv::Mat prev_frame    = cv::imread(fn[img_cnt], 0);
              cv::Mat current_frame = cv::imread(fn[img_cnt+1], 0);

              auto t1 = std::chrono::system_clock::now();
	      //SW TVL1 call- Load two frame and create image pyaramid. TVL1 processsing on current two frames.
	      tvl1->calc(prev_frame,  current_frame , flow);
              auto t2 = std::chrono::system_clock::now();
              auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
              calc_time += value_t1.count();
              calc_cnt++;
   
	      //Generate output image using "flow" matrix
      	      Write_FLowimage( current_frame ,  flow,  img_cnt, out_flow_dir);
           }
	}
	else
	{
           std::cout << "Run with HW accelerator:TVL1 optical flow" << std::endl;
           Mat_<Point2f> flow;
           Ptr<xf::cv::DualTVL1OpticalFlow> tvl1_xfcv = xf::cv::DualTVL1OpticalFlow::create();

           cv::Mat frame0 =  cv::imread(fn[0], 0);
           cv::Mat frame1 =  cv::imread(fn[1], 0);
          
	   //HW TVL1 init
           tvl1_xfcv->init(xclbin_path, frame0.rows, frame0.cols);

	   //HW TVL1 call- Load first two frames and create image pyaramid. No TVL1 processsing
           tvl1_xfcv->calc(frame0, flow);
           tvl1_xfcv->calc(frame1, flow);

	   cv::Mat prev_frame = frame1;  

           for(uint32_t img_cnt=0;img_cnt<fn.size()-1;img_cnt++) {
              cv::Mat current_frame;
              if(img_cnt+2<fn.size())
                 current_frame=  cv::imread(fn[2+img_cnt], 0);
              else
                 current_frame=  cv::imread(fn[0], 0);

              auto t1 = std::chrono::system_clock::now();
	      //HW TVL1 call- Load next frame and create image pyaramid. TVL1 processsing on previous two frames.
              tvl1_xfcv->calc(current_frame, flow);
              auto t2 = std::chrono::system_clock::now();
              auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
              calc_time += value_t1.count();
              calc_cnt++;

	      //Generate output image using "flow" matrix
              Write_FLowimage( prev_frame,  flow,  img_cnt, out_flow_dir);
	      prev_frame = current_frame;
           }   
 	}

    	std::cout << "Total frame: " << fn.size() << "  Performance: " << 1000000.0 / ((float)((calc_time)/calc_cnt)) << " fps\n";

	return 0;
}

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

#define XFPPPre_CONTROL_ADDR_AP_CTRL 			0x00
#define XFPPPre_CONTROL_ADDR_GIE 				0x04
#define XFPPPre_CONTROL_ADDR_IER 				0x08
#define XFPPPre_CONTROL_ADDR_ISR 				0x0c

#define XFPPPre_CONTROL_ADDR_img_inp 			0x10
#define XFPPPre_CONTROL_BITS_img_inp 			64

#define XFPPPre_CONTROL_ADDR_img_out 			0x1C
#define XFPPPre_CONTROL_BITS_img_out 			64

#define XFPPPre_CONTROL_ADDR_params 			0x28
#define XFPPPre_CONTROL_BITS_params 			64

#define XFPPPre_CONTROL_ADDR_in_img_width 		0x34
#define XFPPPre_CONTROL_BITS_in_img_width 		32

#define XFPPPre_CONTROL_ADDR_in_img_height 		0x3C
#define XFPPPre_CONTROL_BITS_in_img_height		32

#define XFPPPre_CONTROL_ADDR_in_img_linestride 	0x44
#define XFPPPre_CONTROL_BITS_in_img_linestride 	32

#define XFPPPre_CONTROL_ADDR_resize_width 		0x4C
#define XFPPPre_CONTROL_BITS_resize_width 		32

#define XFPPPre_CONTROL_ADDR_resize_height 		0x54
#define XFPPPre_CONTROL_BITS_resize_height 		32

#define XFPPPre_CONTROL_ADDR_out_img_width 		0x5C
#define XFPPPre_CONTROL_BITS_out_img_width 		32

#define XFPPPre_CONTROL_ADDR_out_img_height 	0x64
#define XFPPPre_CONTROL_BITS_out_img_height 	32

#define XFPPPre_CONTROL_ADDR_out_img_linestride 0x6C
#define XFPPPre_CONTROL_BITS_out_img_linestride 32

#define XFPPPre_CONTROL_ADDR_roi_posx 			0x74
#define XFPPPre_CONTROL_BITS_roi_posx 			32

#define XFPPPre_CONTROL_ADDR_roi_posy 			0x7C
#define XFPPPre_CONTROL_BITS_roi_posy 			32




/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
//Read and write uncompressed 24 bit BMP image format image
//based on http://en.wikipedia.org/wiki/BMP_file_formt
//Copyright Xilinx


#ifndef __SIMPLE_BMP
#define __SIMPLE_BMP

struct bmpheader_t{
  //Header
  char headerB;
  char headerM;
  uint32_t headerbmpsize;              
  uint16_t headerapp0;
  uint16_t headerapp1;
  uint32_t headerpixelsoffset;

  //DIB header
  uint32_t dibheadersize;
  uint32_t dibwidth;
  uint32_t dibheight;
  uint16_t dibplane;
  uint16_t dibdepth;
  uint32_t dibcompression;
  uint32_t dibsize; 
  uint32_t dibhor;
  uint32_t dibver;
  uint32_t dibpal;
  uint32_t dibimportant;
  
};


struct bmp_t{
  struct bmpheader_t header;
  uint32_t width;
  uint32_t height;
  uint32_t *pixels;
};

int writebmp(char *filename,struct bmp_t *bitmap);

int readbmp(char *filename,struct bmp_t *bitmap);
//-1 file access error
//-2 invalid BMP
//-3 memory allocation error
 

#endif

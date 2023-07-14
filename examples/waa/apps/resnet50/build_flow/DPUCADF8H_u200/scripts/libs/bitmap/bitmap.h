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
#ifndef BITMAP_DOT_H
#define BITMAP_DOT_H

#include <stdlib.h>

class BitmapInterface
{
 private:
  char* core ;
  char* dib ;
  const char* filename ;
  int* image ;

  // Core header information
  unsigned short magicNumber ;
  unsigned int fileSize ;
  unsigned int offsetOfImage ;

  // DIB information
  int sizeOfDIB ;
  int sizeOfImage ;
  int height ;
  int width ;

 public:
  BitmapInterface(const char* f) ;
  ~BitmapInterface() ;

  bool readBitmapFile() ;
  bool writeBitmapFile(int* otherImage = NULL); 

  inline int* bitmap() { return image ; } 
  unsigned int numPixels() { return sizeOfImage/3 ; }

  inline int getHeight() { return height ; }
  inline int getWidth() { return width ; }

} ;

#endif

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
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "bitmap.h"

BitmapInterface::BitmapInterface(const char *f) : filename(f) {
    core = NULL;
    dib = NULL;
    image = NULL;

    magicNumber = 0;
    fileSize = 0;
    offsetOfImage = 0;

    sizeOfDIB = 0;
    sizeOfImage = 0;

    height = -1;
    width = -1;
}

BitmapInterface::~BitmapInterface() {
    if (core != NULL)
        delete[] core;
    if (dib != NULL)
        delete[] dib;
    if (image != NULL)
        delete[] image;
}

bool BitmapInterface::readBitmapFile() {
    // First, open the bitmap file
    int fd;
    unsigned int fileSize;

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Cannot read image file " << filename << std::endl;
        return false;
    }

    core = new char[14];
    read(fd, core, 14);
    magicNumber = (*(unsigned short *)(&(core[0])));
    fileSize = (*(unsigned int *)(&(core[2])));
    offsetOfImage = (*(unsigned int *)(&(core[10])));

    // Just read in the DIB, but don't process it
    sizeOfDIB = offsetOfImage - 14;
    dib = new char[sizeOfDIB];
    read(fd, dib, sizeOfDIB);

    width = (*(int *)(&(dib[4])));
    height = (*(int *)(&(dib[8])));

    sizeOfImage = fileSize - 14 - sizeOfDIB;
    int numPixels = sizeOfImage / 3; // RGB

    image = new int[numPixels];

    for (int i = 0; i < numPixels; ++i) {
        // Use an integer for every pixel even though we might not need that
        //  much space (padding 0 bits in the rest of the integer)
        image[i] = 0;
        read(fd, &(image[i]), 3);
    }

    return true;
}

bool BitmapInterface::writeBitmapFile(int *otherImage) {
    int fd;
    fd = open("output.bmp", O_WRONLY | O_CREAT, 0644);

    if (fd < 0) {
        std::cerr << "Cannot open output.bmp for writing!" << std::endl;
        return false;
    }

    write(fd, core, 14);
    write(fd, dib, sizeOfDIB);

    int numPixels = sizeOfImage / 3;

    int *outputImage = otherImage != NULL ? otherImage : image;

    for (int i = 0; i < numPixels; ++i) {
        write(fd, &(outputImage[i]), 3);
    }

    return true;
}

/*
 * Copyright 2021 Xilinx, Inc.
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
/*-----------------------------
* program to calculate a lookup table of twiddle values
*

* To compile this program run
* gcc -lm -o twiddle.o twiddle.c

* the -lm is necessary to include the math library
*/

#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#define PI 3.14159265358979323846e0
#define PT_SIZE 4096
#define DIR -1

int main() {
    int i, k;
    int tableSizePower;
    int modulus;
    double theta, temp;
    FILE* fp;
    FILE* fp_uut;
    short realshort, imagshort;
    int realint, imagint;
    float realfloat[PT_SIZE / 2], imagfloat[PT_SIZE / 2];

    fp_uut = fopen("../../include/hw/fft_twiddle_lut_dit_cfloat.h", "w");
    fp = fopen("twiddle_master.h", "w");
    double reals[PT_SIZE / 2], imags[PT_SIZE / 2];

    for (i = 0; i < PT_SIZE / 2; i++) { // one octant, then extrapolate from there.
        theta = (double)i * 2.0 * PI / (double)PT_SIZE;
        reals[i] = cos(theta);
        imags[i] = sin(theta) * DIR;
        /*    //use octant symmetry to get second octant
        reals[PT_SIZE/4-i] = imags[i]*DIR;
        imags[PT_SIZE/4-i] = reals[i]*DIR;
        //use octant symmetry to get third octant
        reals[PT_SIZE/4+i] = -reals[PT_SIZE/4-i];
        imags[PT_SIZE/4+i] = imags[PT_SIZE/4-i];
        //use octant symmetry to get fourth octant
        reals[PT_SIZE/2-i] = -reals[i];
        imags[PT_SIZE/2-i] = imags[i];
        */
    }

    // cshort table
    fprintf(fp, "const cint16 twiddle_master[%d] = {\n", PT_SIZE / 2);
    for (i = 0; i < PT_SIZE / 2; i++) {
        temp = round(reals[i] * 32768.0);
        realshort = (short)temp;
        if (temp >= 32767.0) {
            realshort = 32767;
        }
        temp = round(imags[i] * 32768.0);
        imagshort = (short)temp;
        if (temp >= 32768.0) {
            imagshort = 32767;
        }

        fprintf(fp, "{%d, %d}", realshort, imagshort);
        if (i < PT_SIZE / 2 - 1) {
            fprintf(fp, ", ");
        }
        if (i % 8 == 7) {
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "};\n");

    // cint table
    fprintf(fp, "const cint32 twiddle_master[%d] = {\n", PT_SIZE / 2);
    for (i = 0; i < PT_SIZE / 2; i++) {
        temp = round(reals[i] * 2147483648.0);
        realint = (int)temp;
        if (temp >= 2147483647.0) {
            realint = 2147483647;
        }
        temp = round(imags[i] * 2147483648.0);
        imagint = (int)temp;
        if (temp >= 2147483648.0) {
            imagint = 2147483647;
        }

        fprintf(fp, "{%d, %d}", realint, imagint);
        if (i < PT_SIZE / 2 - 1) {
            fprintf(fp, ", ");
        }
        if (i % 8 == 7) {
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "};\n");

    // cfloat table
    fprintf(fp, "const cfloat twiddle_master[%d] = {\n", PT_SIZE / 2);
    for (i = 0; i < PT_SIZE / 2; i++) {
        realfloat[i] = (float)reals[i];
        imagfloat[i] = (float)imags[i];
        fprintf(fp, "{%.9f, %.9f}", realfloat[i], imagfloat[i]);
        if (i < PT_SIZE / 2 - 1) {
            fprintf(fp, ", ");
        }
        if (i % 8 == 7) {
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "};\n");
    fclose(fp);

    fprintf(fp_uut,
            "#ifndef __FFT_TWIDDLE_LUT_DIT_CFLOAT_H__\n#define __FFT_TWIDDLE_LUT_DIT_CFLOAT_H__\n\n#include "
            "\"fft_com_inc.h\"\n");
    fprintf(fp_uut, "// DO NOT HAND EDIT THIS FILE. IT WAS CREATED using ../tests/inc/twiddle.c\n\n");

    for (tableSizePower = 11; tableSizePower >= 0; tableSizePower--) {
        k = 0;
        fprintf(fp_uut, "const cfloat chess_storage(%%chess_alignof(v4cint16)) fft_lut_tw%d_cfloat[%d] = {\n",
                (1 << tableSizePower), (1 << tableSizePower));
        for (i = 0; i < PT_SIZE / 2; i += (1 << (11 - tableSizePower))) {
            fprintf(fp_uut, "{%.9f, %.9f}", realfloat[i], imagfloat[i]);
            if (i < PT_SIZE / 2 - (1 << (11 - tableSizePower))) {
                fprintf(fp_uut, ", ");
            }
            if ((++k) == 8) {
                fprintf(fp_uut, "\n");
                k = 0;
            }
        }
        fprintf(fp_uut, "};\n");
    } // for tableSizePower
    fprintf(fp_uut, "#endif //__FFT_TWIDDLE_LUT_DIT_CFLOAT_H__\n");
    fclose(fp_uut);
}

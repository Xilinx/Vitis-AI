/*
 * Copyright (c) 2019 Mujib Haider
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "jfif.h"

const jfif_quantization_table_t jfif_default_q_table[2] = {
    // Luma Table
    {
        .DQT = 65499,
        .Lq  = 67,
        .Pq  = 0,
        .Tq  = 0,
        .Qk  = {
            16,  11,  12,  14,  12,  10,  16,  14,
            13,  14,  18,  17,  16,  19,  24,  40,
            26,  24,  22,  22,  24,  49,  35,  37,
            29,  40,  58,  51,  61,  60,  57,  51,
            56,  55,  64,  72,  92,  78,  64,  68,
            87,  69,  55,  56,  80,  109, 81,  87,
            95,  98,  103, 104, 103, 62,  77,  113,
            121, 112, 100, 120, 92,  101, 103, 99            
        }
    },

    // Chroma Table
    {
        .DQT = 65499,
        .Lq  = 67,
        .Pq  = 0,
        .Tq  = 1,
        .Qk  = {
            17,  18,  18,  24,  21,  24,  47,  26,
            26,  47,  99,  66,  56,  66,  99,  99,
            99,  99,  99,  99,  99,  99,  99,  99,
            99,  99,  99,  99,  99,  99,  99,  99,
            99,  99,  99,  99,  99,  99,  99,  99,
            99,  99,  99,  99,  99,  99,  99,  99,
            99,  99,  99,  99,  99,  99,  99,  99,
            99,  99,  99,  99,  99,  99,  99,  99            
        }
    }
};

const jfif_huffman_table_t jfif_default_h_table[4] = {
    // Luma DC Table
    {
        .DHT = 65476,
        .Lh  = 31,
        .Tc  = 0,
        .Th  = 0,
        .L   = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
        .V   = {
            {},
            {0},
            {1, 2, 3, 4, 5},
            {6},
            {7},
            {8},
            {9},
            {10},
            {11},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        }
    },

    // Luma AC Table
    {
        .DHT = 65476,
        .Lh  = 181,
        .Tc  = 1,
        .Th  = 0,
        .L   = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125},
        .V   = {
            {},
            {1, 2},
            {3},
            {0, 4, 17},
            {5, 18, 33},
            {49, 65},
            {6, 19, 81, 97},
            {7, 34, 113},
            {20, 50, 129, 145, 161},
            {8, 35, 66, 177, 193},
            {21, 82, 209, 240},
            {36, 51, 98, 114},
            {},
            {},
            {130},
            {9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250},
        }
    },

    // Chroma DC Table
    {
        .DHT = 65476,
        .Lh  = 31,
        .Tc  = 0,
        .Th  = 1,
        .L   = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
        .V   = {
            {},
            {0, 1, 2},
            {3},
            {4},
            {5},
            {6},
            {7},
            {8},
            {9},
            {10},
            {11},
            {},
            {},
            {},
            {},
            {},
        }
    },

    // Chroma AC Table
    {
        .DHT = 65476,
        .Lh  = 181,
        .Tc  = 1,
        .Th  = 1,
        .L   = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119},
        .V   = {
            {},
            {0, 1},
            {2},
            {3, 17},
            {4, 5, 33, 49},
            {6, 18, 65, 81},
            {7, 97, 113},
            {19, 34, 50, 129},
            {8, 20, 66, 145, 161, 177, 193},
            {9, 35, 51, 82, 240},
            {21, 98, 114, 209},
            {10, 22, 36, 52},
            {},
            {255},
            {37, 241},
            {23, 24, 25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248, 249, 250},
        }
    }
};
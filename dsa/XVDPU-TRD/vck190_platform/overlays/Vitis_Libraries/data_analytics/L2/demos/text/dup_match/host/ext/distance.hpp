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

/*
 * MIT License
 *
 * Copyright (c) 2016 Forest Gregg and DataMade
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

/*
 * This file comes from open source dedupeio/affinegap: https://github.com/dedupeio/affinegap.
 * This file is used by XILINX.
 * In order to be called, convert it into C++ language by XILINX.
 */

#ifndef _EXT_DISTANCE_HPP_
#define _EXT_DISTANCE_HPP_

#include <vector>
#include <cstdint>
#include <string>

namespace ext {
static float min2(float a, float b) {
    if (a < b)
        return a;
    else
        return b;
}

static float min3(float a, float b, float c) {
    return min2(min2(a, b), c);
}

static float wholeFieldDistance(std::string& string1, std::string& string2) {
    if (string1 == string2)
        return 1.0;
    else
        return 0.0;
}

static float affineGapDistance(std::string& string1,
                               std::string& string2,
                               float matchWeight = 1,
                               float mismatchWeight = 11,
                               float gapWeight = 10,
                               float spaceWeight = 7,
                               float abbreviation_scale = 0.125) {
    /*
    Calculate the affine gap distance between two strings

    Default weights are from Alvaro Monge and Charles Elkan, 1996,
    "The field matching problem: Algorithms and applications"
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.23.9685
    */

    int length1 = string1.size();
    int length2 = string2.size();
    std::string string_a = string1;
    std::string string_b = string2;

    if (string_a == string_b && (matchWeight == min3(matchWeight, mismatchWeight, gapWeight)))
        return matchWeight * length1;

    if (length1 < length2) {
        length1 = string2.size();
        length2 = string1.size();
        string_a = string2;
        string_b = string1;
    }

    // Initialize C Arrays
    int memory_size = sizeof(float) * (length1 + 1);
    float* D = (float*)malloc(memory_size);
    float* V_current = (float*)malloc(memory_size);
    float* V_previous = (float*)malloc(memory_size);

    int i, j;
    float distance;

    // Set up Recurrence relations
    //
    // Base conditions
    // V(0, 0) = 0
    // V(0, j) = gapWeight + spaceWeight * i
    // D(0, j) = Infinity
    V_current[0] = 0;
    for (int j = 1; j < length1 + 1; j++) {
        V_current[j] = gapWeight + spaceWeight * j;
        D[j] = 0x3FFFFFFF;
    }

    for (int i = 1; i < length2 + 1; i++) {
        char char2 = string_b[i - 1];
        // V_previous = V_current
        for (int x = 0; x < length1 + 1; x++) V_previous[x] = V_current[x];

        // Base conditions
        // V(i, 0) = gapWeight + spaceWeight * i
        // I(i, 0) = Infinity
        V_current[0] = gapWeight + spaceWeight * i;
        float I = 0x3FFFFFFF;

        for (int j = 1; j < length1 + 1; j++) {
            char char1 = string_a[j - 1];

            // I(i, j) is the edit distance if the jth character of string 1
            // was inserted into string 2.
            //
            // I(i, j) = min(I(i, j - 1), V(i, j - 1) + gapWeight) + spaceWeight
            if (j <= length2)
                I = min2(I, V_current[j - 1] + gapWeight) + spaceWeight;
            else {
                // Pay less for abbreviations
                // i.e.'spago (los angeles) to ' spago'
                I = (min2(I, V_current[j - 1] + gapWeight * abbreviation_scale) + spaceWeight * abbreviation_scale);
            }

            // D(i, j) is the edit distance if the ith character of string 2
            // was deleted from string 1
            //
            // D(i, j) = min((i - 1, j), V(i - 1, j) + gapWeight) + spaceWeight
            D[j] = min2(D[j], V_previous[j] + gapWeight) + spaceWeight;

            // M(i, j) is the edit distance if the ith and jth characters
            // match or mismatch
            //
            // M(i, j) = V(i - 1, j - 1) + (matchWeight | misMatchWeight)
            float M;
            if (char2 == char1)
                M = V_previous[j - 1] + matchWeight;
            else
                M = V_previous[j - 1] + mismatchWeight;

            // V(i, j) is the minimum edit distance
            //
            // V(i, j) = min(E(i, j), F(i, j), G(i, j))
            V_current[j] = min3(I, D[j], M);
        }
    }

    distance = V_current[length1];

    free(D);
    free(V_current);
    free(V_previous);

    return distance;
}

static float normalizedAffineGapDistance(std::string& string1,
                                         std::string& string2,
                                         float matchWeight = 1,
                                         float mismatchWeight = 11,
                                         float gapWeight = 10,
                                         float spaceWeight = 7,
                                         float abbreviation_scale = .125) {
    float normalizer = string1.size() + string2.size();

    float distance =
        affineGapDistance(string1, string2, matchWeight, mismatchWeight, gapWeight, spaceWeight, abbreviation_scale);

    return distance / normalizer;
}

static float distance(std::string& string1, std::string& string2, bool flag) {
    if (flag)
        return wholeFieldDistance(string1, string2);
    else
        return normalizedAffineGapDistance(string1, string2);
}

} // namespace ext

#endif

/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2003, 2004 Ferdinando Ametrano
 Copyright (C) 2006 Richard Gould
 Copyright (C) 2007 Mark Joshi

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file sobolrsg.hpp
    \brief Sobol low-discrepancy sequence generator
*/

// ===========================================================================
// NOTE: The following copyright notice applies to the original code,
//
// Copyright (C) 2002 Peter J�ckel "Monte Carlo Methods in Finance".
// All rights reserved.
//
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// ===========================================================================

/* This file comes from open source QuantLib, https://github.com/lballabio/QuantLib.git. The license for it is BSD-3-Clause.
 * This file is used by XILINX as a reference.
 * In order to simplify its usage, some modification is added to the source file, such as change the data type, remove some not used codes.
 * The modification is done by XILINX.
 */

#ifndef quantlib_sobol_ld_rsg_hpp
#define quantlib_sobol_ld_rsg_hpp


#include <cmath>
#include <vector>

namespace QuantLib {

    //! Sobol low-discrepancy sequence generator
    /*! A Gray code counter and bitwise operations are used for very
        fast sequence generation.

        The implementation relies on primitive polynomials modulo two
        from the book "Monte Carlo Methods in Finance" by Peter
        Jäckel.

        21 200 primitive polynomials modulo two are provided in QuantLib.
        Jäckel has calculated 8 129 334 polynomials: if you need that many
        dimensions you can replace the primitivepolynomials.cpp file included
        in QuantLib with the one provided in the CD of the "Monte Carlo
        Methods in Finance" book.

        The choice of initialization numbers (also know as free direction
        integers) is crucial for the homogeneity properties of the sequence.
        Sobol defines two homogeneity properties: Property A and Property A'.

        The unit initialization numbers suggested in "Numerical
        Recipes in C", 2nd edition, by Press, Teukolsky, Vetterling,
        and Flannery (section 7.7) fail the test for Property A even
        for low dimensions.

        Bratley and Fox published coefficients of the free direction
        integers up to dimension 40, crediting unpublished work of
        Sobol' and Levitan. See Bratley, P., Fox, B.L. (1988)
        "Algorithm 659: Implementing Sobol's quasirandom sequence
        generator," ACM Transactions on Mathematical Software
        14:88-100. These values satisfy Property A for d<=20 and d =
        23, 31, 33, 34, 37; Property A' holds for d<=6.

        Jäckel provides in his book (section 8.3) initialization
        numbers up to dimension 32. Coefficients for d<=8 are the same
        as in Bradley-Fox, so Property A' holds for d<=6 but Property
        A holds for d<=32.

        The implementation of Lemieux, Cieslak, and Luttmer includes
        coefficients of the free direction integers up to dimension
        360.  Coefficients for d<=40 are the same as in Bradley-Fox.
        For dimension 40<d<=360 the coefficients have
        been calculated as optimal values based on the "resolution"
        criterion. See "RandQMC user's guide - A package for
        randomized quasi-Monte Carlo methods in C," by C. Lemieux,
        M. Cieslak, and K. Luttmer, version January 13 2004, and
        references cited there
        (http://www.math.ucalgary.ca/~lemieux/randqmc.html).
        The values up to d<=360 has been provided to the QuantLib team by
        Christiane Lemieux, private communication, September 2004.

        For more info on Sobol' sequences see also "Monte Carlo
        Methods in Financial Engineering," by P. Glasserman, 2004,
        Springer, section 5.2.3

        The Joe--Kuo numbers and the Kuo numbers are due to Stephen Joe
        and Frances Kuo.

        S. Joe and F. Y. Kuo, Constructing Sobol sequences with better
        two-dimensional projections, preprint Nov 22 2007

        See http://web.maths.unsw.edu.au/~fkuo/sobol/ for more information.

        The Joe-Kuo numbers are available under a BSD-style license
        available at the above link.

        Note that the Kuo numbers were generated to work with a
        different ordering of primitive polynomials for the first 40
        or so dimensions which is why we have the Alternative
        Primitive Polynomials.

        \test
        - the correctness of the returned values is tested by
          reproducing known good values.
        - the correctness of the returned values is tested by checking
          their discrepancy against known good values.
    */


// number of dimensions in the alternative primitive polynomials
const unsigned int maxAltDegree = 128;

static const long PrimitivePolynomialDegree01[]={
0, /* x+1 (1)(1) */
-1 };

static const long PrimitivePolynomialDegree02[]={
1, /* x^2+x+1 (1)1(1) */
-1 };

static const long PrimitivePolynomialDegree03[]={
1, /* x^3    +x+1 (1)01(1) */
2, /* x^3+x^2  +1 (1)10(1) */
-1 };

static const long PrimitivePolynomialDegree04[]={
1, /* x^4+       +x+1 (1)001(1) */
4, /* x^4+x^3+     +1 (1)100(1) */
-1 };

static const long PrimitivePolynomialDegree05[]={
2,  /* x^5        +x^2  +1 (1)0010(1) */
4,  /* x^5    +x^3      +1 (1)0100(1) */
7,  /* x^5    +x^3+x^2+x+1 (1)0111(1) */
11, /* x^5+x^4    +x^2+x+1 (1)1011(1) */
13, /* x^5+x^4+x^3    +x+1 (1)1101(1) */
14, /* x^5+x^4+x^3+x^2  +1 (1)1110(1) */
-1 };

static const long PrimitivePolynomialDegree06[]={
1,  /* x^6                +x+1 (1)00001(1) */
13, /* x^6    +x^4+x^3    +x+1 (1)01101(1) */
16, /* x^6+x^5              +1 (1)10000(1) */
19, /* x^6            +x^2+x+1 (1)10011(1) */
22, /* x^6+x^5    +x^3+x^2  +1 (1)10110(1) */
25, /* x^6+x^5+x^4        +x+1 (1)11001(1) */
-1 };

static const long PrimitivePolynomialDegree07[]={
1,  /* x^7                    +x+1 (1)000001(1) */
4,  /* x^7            +x^3      +1 (1)000100(1) */
7,  /* x^7            +x^3+x^2+x+1 (1)000111(1) */
8,  /* x^7        +x^4          +1 (1)001000(1) */
14, /* x^7        +x^4+x^3+x^2  +1 (1)001110(1) */
19, /* x^7    +x^5        +x^2+x+1 (1)010011(1) */
21, /* x^7    +x^5    +x^3    +x+1 (1)010101(1) */
28, /* x^7    +x^5+x^4+x^3      +1 (1)011100(1) */
31, /* x^7    +x^5+x^4+x^3+x^2+x+1 (1)011111(1) */
32, /* x^7+x^6                  +1 (1)100000(1) */
37, /* x^7+x^6        +x^3    +x+1 (1)100101(1) */
41, /* x^7+x^6    +x^4        +x+1 (1)101001(1) */
42, /* x^7+x^6    +x^4    +x^2  +1 (1)101010(1) */
/* 32 polynomials so far ... let's go ahead */
50, /* x^7+x^6+x^5        +x^2  +1 (1)110010(1) */
55, /* x^7+x^6+x^5    +x^3+x^2+x+1 (1)110111(1) */
56, /* x^7+x^6+x^5+x^4          +1 (1)111000(1) */
59, /* x^7+x^6+x^5+x^4    +x^2+x+1 (1)111011(1) */
62, /* x^7+x^6+x^5+x^4+x^3+x^2  +1 (1)111110(1) */
-1 };

static const long PrimitivePolynomialDegree08[]={
14,
21,
22,
38,
47,
49,
50,
52,
56,
67,
70,
84,
97,
103,
115,
122,
-1 };

static const long PrimitivePolynomialDegree09[]={
8,
13,
16,
22,
25,
44,
47,
52,
55,
59,
62,
67,
74,
81,
82,
87,
91,
94,
103,
104,
109,
122,
124,
137,
138,
143,
145,
152,
157,
167,
173,
176,
181,
182,
185,
191,
194,
199,
218,
220,
227,
229,
230,
234,
236,
241,
244,
253,
-1 };

static const long PrimitivePolynomialDegree10[]={
4,
13,
19,
22,
50,
55,
64,
69,
98,
107,
115,
121,
127,
134,
140,
145,
152,
158,
161,
171,
181,
194,
199,
203,
208,
227,
242,
251,
253,
265,
266,
274,
283,
289,
295,
301,
316,
319,
324,
346,
352,
361,
367,
382,
395,
398
};

#define N_ALT_MAX_DEGREE 10

const long *const AltPrimitivePolynomials[N_ALT_MAX_DEGREE]=
{
PrimitivePolynomialDegree01,
PrimitivePolynomialDegree02,
PrimitivePolynomialDegree03,
PrimitivePolynomialDegree04,
PrimitivePolynomialDegree05,
PrimitivePolynomialDegree06,
PrimitivePolynomialDegree07,
PrimitivePolynomialDegree08,
PrimitivePolynomialDegree09,
PrimitivePolynomialDegree10
};

const unsigned int dim1JoeKuoD6Init[]   =   { 1, 0 };
const unsigned int dim2JoeKuoD6Init[]   =   { 1, 3, 0 };
const unsigned int dim3JoeKuoD6Init[]   =   { 1, 3, 1, 0 };
const unsigned int dim4JoeKuoD6Init[]   =   { 1, 1, 1, 0 };
const unsigned int dim5JoeKuoD6Init[]   =   { 1, 1, 3, 3, 0 };
const unsigned int dim6JoeKuoD6Init[]   =   { 1, 3, 5, 13, 0 };
const unsigned int dim7JoeKuoD6Init[]   =   { 1, 1, 5, 5, 17, 0 };
const unsigned int dim8JoeKuoD6Init[]   =   { 1, 1, 5, 5, 5, 0 };
const unsigned int dim9JoeKuoD6Init[]   =   { 1, 1, 7, 11, 19, 0 };
const unsigned int dim10JoeKuoD6Init[]   =   { 1, 1, 5, 1, 1, 0 };
const unsigned int dim11JoeKuoD6Init[]   =   { 1, 1, 1, 3, 11, 0 };
const unsigned int dim12JoeKuoD6Init[]   =   { 1, 3, 5, 5, 31, 0 };
const unsigned int dim13JoeKuoD6Init[]   =   { 1, 3, 3, 9, 7, 49, 0 };
const unsigned int dim14JoeKuoD6Init[]   =   { 1, 1, 1, 15, 21, 21, 0 };
const unsigned int dim15JoeKuoD6Init[]   =   { 1, 3, 1, 13, 27, 49, 0 };
const unsigned int dim16JoeKuoD6Init[]   =   { 1, 1, 1, 15, 7, 5, 0 };
const unsigned int dim17JoeKuoD6Init[]   =   { 1, 3, 1, 15, 13, 25, 0 };
const unsigned int dim18JoeKuoD6Init[]   =   { 1, 1, 5, 5, 19, 61, 0 };
const unsigned int dim19JoeKuoD6Init[]   =   { 1, 3, 7, 11, 23, 15, 103, 0 };
const unsigned int dim20JoeKuoD6Init[]   =   { 1, 3, 7, 13, 13, 15, 69, 0 };
const unsigned int dim21JoeKuoD6Init[]   =   { 1, 1, 3, 13, 7, 35, 63, 0 };
const unsigned int dim22JoeKuoD6Init[]   =   { 1, 3, 5, 9, 1, 25, 53, 0 };
const unsigned int dim23JoeKuoD6Init[]   =   { 1, 3, 1, 13, 9, 35, 107, 0 };
const unsigned int dim24JoeKuoD6Init[]   =   { 1, 3, 1, 5, 27, 61, 31, 0 };
const unsigned int dim25JoeKuoD6Init[]   =   { 1, 1, 5, 11, 19, 41, 61, 0 };
const unsigned int dim26JoeKuoD6Init[]   =   { 1, 3, 5, 3, 3, 13, 69, 0 };
const unsigned int dim27JoeKuoD6Init[]   =   { 1, 1, 7, 13, 1, 19, 1, 0 };
const unsigned int dim28JoeKuoD6Init[]   =   { 1, 3, 7, 5, 13, 19, 59, 0 };
const unsigned int dim29JoeKuoD6Init[]   =   { 1, 1, 3, 9, 25, 29, 41, 0 };
const unsigned int dim30JoeKuoD6Init[]   =   { 1, 3, 5, 13, 23, 1, 55, 0 };
const unsigned int dim31JoeKuoD6Init[]   =   { 1, 3, 7, 3, 13, 59, 17, 0 };
const unsigned int dim32JoeKuoD6Init[]   =   { 1, 3, 1, 3, 5, 53, 69, 0 };
const unsigned int dim33JoeKuoD6Init[]   =   { 1, 1, 5, 5, 23, 33, 13, 0 };
const unsigned int dim34JoeKuoD6Init[]   =   { 1, 1, 7, 7, 1, 61, 123, 0 };
const unsigned int dim35JoeKuoD6Init[]   =   { 1, 1, 7, 9, 13, 61, 49, 0 };
const unsigned int dim36JoeKuoD6Init[]   =   { 1, 3, 3, 5, 3, 55, 33, 0 };
const unsigned int dim37JoeKuoD6Init[]   =   { 1, 3, 1, 15, 31, 13, 49, 245, 0 };
const unsigned int dim38JoeKuoD6Init[]   =   { 1, 3, 5, 15, 31, 59, 63, 97, 0 };
const unsigned int dim39JoeKuoD6Init[]   =   { 1, 3, 1, 11, 11, 11, 77, 249, 0 };
const unsigned int dim40JoeKuoD6Init[]   =   { 1, 3, 1, 11, 27, 43, 71, 9, 0 };
const unsigned int dim41JoeKuoD6Init[]   =   { 1, 1, 7, 15, 21, 11, 81, 45, 0 };
const unsigned int dim42JoeKuoD6Init[]   =   { 1, 3, 7, 3, 25, 31, 65, 79, 0 };
const unsigned int dim43JoeKuoD6Init[]   =   { 1, 3, 1, 1, 19, 11, 3, 205, 0 };
const unsigned int dim44JoeKuoD6Init[]   =   { 1, 1, 5, 9, 19, 21, 29, 157, 0 };
const unsigned int dim45JoeKuoD6Init[]   =   { 1, 3, 7, 11, 1, 33, 89, 185, 0 };
const unsigned int dim46JoeKuoD6Init[]   =   { 1, 3, 3, 3, 15, 9, 79, 71, 0 };
const unsigned int dim47JoeKuoD6Init[]   =   { 1, 3, 7, 11, 15, 39, 119, 27, 0 };
const unsigned int dim48JoeKuoD6Init[]   =   { 1, 1, 3, 1, 11, 31, 97, 225, 0 };
const unsigned int dim49JoeKuoD6Init[]   =   { 1, 1, 1, 3, 23, 43, 57, 177, 0 };
const unsigned int dim50JoeKuoD6Init[]   =   { 1, 3, 7, 7, 17, 17, 37, 71, 0 };
const unsigned int dim51JoeKuoD6Init[]   =   { 1, 3, 1, 5, 27, 63, 123, 213, 0 };
const unsigned int dim52JoeKuoD6Init[]   =   { 1, 1, 3, 5, 11, 43, 53, 133, 0 };
const unsigned int dim53JoeKuoD6Init[]   =   { 1, 3, 5, 5, 29, 17, 47, 173, 479, 0 };
const unsigned int dim54JoeKuoD6Init[]   =   { 1, 3, 3, 11, 3, 1, 109, 9, 69, 0 };
const unsigned int dim55JoeKuoD6Init[]   =   { 1, 1, 1, 5, 17, 39, 23, 5, 343, 0 };
const unsigned int dim56JoeKuoD6Init[]   =   { 1, 3, 1, 5, 25, 15, 31, 103, 499, 0 };
const unsigned int dim57JoeKuoD6Init[]   =   { 1, 1, 1, 11, 11, 17, 63, 105, 183, 0 };
const unsigned int dim58JoeKuoD6Init[]   =   { 1, 1, 5, 11, 9, 29, 97, 231, 363, 0 };
const unsigned int dim59JoeKuoD6Init[]   =   { 1, 1, 5, 15, 19, 45, 41, 7, 383, 0 };
const unsigned int dim60JoeKuoD6Init[]   =   { 1, 3, 7, 7, 31, 19, 83, 137, 221, 0 };
const unsigned int dim61JoeKuoD6Init[]   =   { 1, 1, 1, 3, 23, 15, 111, 223, 83, 0 };
const unsigned int dim62JoeKuoD6Init[]   =   { 1, 1, 5, 13, 31, 15, 55, 25, 161, 0 };
const unsigned int dim63JoeKuoD6Init[]   =   { 1, 1, 3, 13, 25, 47, 39, 87, 257, 0 };
const unsigned int dim64JoeKuoD6Init[]   =   { 1, 1, 1, 11, 21, 53, 125, 249, 293, 0 };
const unsigned int dim65JoeKuoD6Init[]   =   { 1, 1, 7, 11, 11, 7, 57, 79, 323, 0 };
const unsigned int dim66JoeKuoD6Init[]   =   { 1, 1, 5, 5, 17, 13, 81, 3, 131, 0 };
const unsigned int dim67JoeKuoD6Init[]   =   { 1, 1, 7, 13, 23, 7, 65, 251, 475, 0 };
const unsigned int dim68JoeKuoD6Init[]   =   { 1, 3, 5, 1, 9, 43, 3, 149, 11, 0 };
const unsigned int dim69JoeKuoD6Init[]   =   { 1, 1, 3, 13, 31, 13, 13, 255, 487, 0 };
const unsigned int dim70JoeKuoD6Init[]   =   { 1, 3, 3, 1, 5, 63, 89, 91, 127, 0 };
const unsigned int dim71JoeKuoD6Init[]   =   { 1, 1, 3, 3, 1, 19, 123, 127, 237, 0 };
const unsigned int dim72JoeKuoD6Init[]   =   { 1, 1, 5, 7, 23, 31, 37, 243, 289, 0 };
const unsigned int dim73JoeKuoD6Init[]   =   { 1, 1, 5, 11, 17, 53, 117, 183, 491, 0 };
const unsigned int dim74JoeKuoD6Init[]   =   { 1, 1, 1, 5, 1, 13, 13, 209, 345, 0 };
const unsigned int dim75JoeKuoD6Init[]   =   { 1, 1, 3, 15, 1, 57, 115, 7, 33, 0 };
const unsigned int dim76JoeKuoD6Init[]   =   { 1, 3, 1, 11, 7, 43, 81, 207, 175, 0 };
const unsigned int dim77JoeKuoD6Init[]   =   { 1, 3, 1, 1, 15, 27, 63, 255, 49, 0 };
const unsigned int dim78JoeKuoD6Init[]   =   { 1, 3, 5, 3, 27, 61, 105, 171, 305, 0 };
const unsigned int dim79JoeKuoD6Init[]   =   { 1, 1, 5, 3, 1, 3, 57, 249, 149, 0 };
const unsigned int dim80JoeKuoD6Init[]   =   { 1, 1, 3, 5, 5, 57, 15, 13, 159, 0 };
const unsigned int dim81JoeKuoD6Init[]   =   { 1, 1, 1, 11, 7, 11, 105, 141, 225, 0 };
const unsigned int dim82JoeKuoD6Init[]   =   { 1, 3, 3, 5, 27, 59, 121, 101, 271, 0 };
const unsigned int dim83JoeKuoD6Init[]   =   { 1, 3, 5, 9, 11, 49, 51, 59, 115, 0 };
const unsigned int dim84JoeKuoD6Init[]   =   { 1, 1, 7, 1, 23, 45, 125, 71, 419, 0 };
const unsigned int dim85JoeKuoD6Init[]   =   { 1, 1, 3, 5, 23, 5, 105, 109, 75, 0 };
const unsigned int dim86JoeKuoD6Init[]   =   { 1, 1, 7, 15, 7, 11, 67, 121, 453, 0 };
const unsigned int dim87JoeKuoD6Init[]   =   { 1, 3, 7, 3, 9, 13, 31, 27, 449, 0 };
const unsigned int dim88JoeKuoD6Init[]   =   { 1, 3, 1, 15, 19, 39, 39, 89, 15, 0 };
const unsigned int dim89JoeKuoD6Init[]   =   { 1, 1, 1, 1, 1, 33, 73, 145, 379, 0 };
const unsigned int dim90JoeKuoD6Init[]   =   { 1, 3, 1, 15, 15, 43, 29, 13, 483, 0 };
const unsigned int dim91JoeKuoD6Init[]   =   { 1, 1, 7, 3, 19, 27, 85, 131, 431, 0 };
const unsigned int dim92JoeKuoD6Init[]   =   { 1, 3, 3, 3, 5, 35, 23, 195, 349, 0 };
const unsigned int dim93JoeKuoD6Init[]   =   { 1, 3, 3, 7, 9, 27, 39, 59, 297, 0 };
const unsigned int dim94JoeKuoD6Init[]   =   { 1, 1, 3, 9, 11, 17, 13, 241, 157, 0 };
const unsigned int dim95JoeKuoD6Init[]   =   { 1, 3, 7, 15, 25, 57, 33, 189, 213, 0 };
const unsigned int dim96JoeKuoD6Init[]   =   { 1, 1, 7, 1, 9, 55, 73, 83, 217, 0 };
const unsigned int dim97JoeKuoD6Init[]   =   { 1, 3, 3, 13, 19, 27, 23, 113, 249, 0 };
const unsigned int dim98JoeKuoD6Init[]   =   { 1, 3, 5, 3, 23, 43, 3, 253, 479, 0 };
const unsigned int dim99JoeKuoD6Init[]   =   { 1, 1, 5, 5, 11, 5, 45, 117, 217, 0 };
const unsigned int dim100JoeKuoD6Init[]   =   { 1, 3, 3, 7, 29, 37, 33, 123, 147, 0 };
const unsigned int dim101JoeKuoD6Init[]   =   { 1, 3, 1, 15, 5, 5, 37, 227, 223, 459, 0 };
const unsigned int dim102JoeKuoD6Init[]   =   { 1, 1, 7, 5, 5, 39, 63, 255, 135, 487, 0 };
const unsigned int dim103JoeKuoD6Init[]   =   { 1, 3, 1, 7, 9, 7, 87, 249, 217, 599, 0 };
const unsigned int dim104JoeKuoD6Init[]   =   { 1, 1, 3, 13, 9, 47, 7, 225, 363, 247, 0 };
const unsigned int dim105JoeKuoD6Init[]   =   { 1, 3, 7, 13, 19, 13, 9, 67, 9, 737, 0 };
const unsigned int dim106JoeKuoD6Init[]   =   { 1, 3, 5, 5, 19, 59, 7, 41, 319, 677, 0 };
const unsigned int dim107JoeKuoD6Init[]   =   { 1, 1, 5, 3, 31, 63, 15, 43, 207, 789, 0 };
const unsigned int dim108JoeKuoD6Init[]   =   { 1, 1, 7, 9, 13, 39, 3, 47, 497, 169, 0 };
const unsigned int dim109JoeKuoD6Init[]   =   { 1, 3, 1, 7, 21, 17, 97, 19, 415, 905, 0 };
const unsigned int dim110JoeKuoD6Init[]   =   { 1, 3, 7, 1, 3, 31, 71, 111, 165, 127, 0 };
const unsigned int dim111JoeKuoD6Init[]   =   { 1, 1, 5, 11, 1, 61, 83, 119, 203, 847, 0 };
const unsigned int dim112JoeKuoD6Init[]   =   { 1, 3, 3, 13, 9, 61, 19, 97, 47, 35, 0 };
const unsigned int dim113JoeKuoD6Init[]   =   { 1, 1, 7, 7, 15, 29, 63, 95, 417, 469, 0 };
const unsigned int dim114JoeKuoD6Init[]   =   { 1, 3, 1, 9, 25, 9, 71, 57, 213, 385, 0 };
const unsigned int dim115JoeKuoD6Init[]   =   { 1, 3, 5, 13, 31, 47, 101, 57, 39, 341, 0 };
const unsigned int dim116JoeKuoD6Init[]   =   { 1, 1, 3, 3, 31, 57, 125, 173, 365, 551, 0 };
const unsigned int dim117JoeKuoD6Init[]   =   { 1, 3, 7, 1, 13, 57, 67, 157, 451, 707, 0 };
const unsigned int dim118JoeKuoD6Init[]   =   { 1, 1, 1, 7, 21, 13, 105, 89, 429, 965, 0 };
const unsigned int dim119JoeKuoD6Init[]   =   { 1, 1, 5, 9, 17, 51, 45, 119, 157, 141, 0 };
const unsigned int dim120JoeKuoD6Init[]   =   { 1, 3, 7, 7, 13, 45, 91, 9, 129, 741, 0 };
const unsigned int dim121JoeKuoD6Init[]   =   { 1, 3, 7, 1, 23, 57, 67, 141, 151, 571, 0 };
const unsigned int dim122JoeKuoD6Init[]   =   { 1, 1, 3, 11, 17, 47, 93, 107, 375, 157, 0 };
const unsigned int dim123JoeKuoD6Init[]   =   { 1, 3, 3, 5, 11, 21, 43, 51, 169, 915, 0 };
const unsigned int dim124JoeKuoD6Init[]   =   { 1, 1, 5, 3, 15, 55, 101, 67, 455, 625, 0 };
const unsigned int dim125JoeKuoD6Init[]   =   { 1, 3, 5, 9, 1, 23, 29, 47, 345, 595, 0 };
const unsigned int dim126JoeKuoD6Init[]   =   { 1, 3, 7, 7, 5, 49, 29, 155, 323, 589, 0 };
const unsigned int dim127JoeKuoD6Init[]   =   { 1, 3, 3, 7, 5, 41, 127, 61, 261, 717, 0 };
const unsigned int dim128JoeKuoD6Init[]   =   { 1, 3, 7, 7, 17, 23, 117, 67, 129, 1009, 0 };

const unsigned int * const JoeKuoD6initializers[21200]
=
{
dim1JoeKuoD6Init,
dim2JoeKuoD6Init,
dim3JoeKuoD6Init,
dim4JoeKuoD6Init,
dim5JoeKuoD6Init,
dim6JoeKuoD6Init,
dim7JoeKuoD6Init,
dim8JoeKuoD6Init,
dim9JoeKuoD6Init,
dim10JoeKuoD6Init,
dim11JoeKuoD6Init,
dim12JoeKuoD6Init,
dim13JoeKuoD6Init,
dim14JoeKuoD6Init,
dim15JoeKuoD6Init,
dim16JoeKuoD6Init,
dim17JoeKuoD6Init,
dim18JoeKuoD6Init,
dim19JoeKuoD6Init,
dim20JoeKuoD6Init,
dim21JoeKuoD6Init,
dim22JoeKuoD6Init,
dim23JoeKuoD6Init,
dim24JoeKuoD6Init,
dim25JoeKuoD6Init,
dim26JoeKuoD6Init,
dim27JoeKuoD6Init,
dim28JoeKuoD6Init,
dim29JoeKuoD6Init,
dim30JoeKuoD6Init,
dim31JoeKuoD6Init,
dim32JoeKuoD6Init,
dim33JoeKuoD6Init,
dim34JoeKuoD6Init,
dim35JoeKuoD6Init,
dim36JoeKuoD6Init,
dim37JoeKuoD6Init,
dim38JoeKuoD6Init,
dim39JoeKuoD6Init,
dim40JoeKuoD6Init,
dim41JoeKuoD6Init,
dim42JoeKuoD6Init,
dim43JoeKuoD6Init,
dim44JoeKuoD6Init,
dim45JoeKuoD6Init,
dim46JoeKuoD6Init,
dim47JoeKuoD6Init,
dim48JoeKuoD6Init,
dim49JoeKuoD6Init,
dim50JoeKuoD6Init,
dim51JoeKuoD6Init,
dim52JoeKuoD6Init,
dim53JoeKuoD6Init,
dim54JoeKuoD6Init,
dim55JoeKuoD6Init,
dim56JoeKuoD6Init,
dim57JoeKuoD6Init,
dim58JoeKuoD6Init,
dim59JoeKuoD6Init,
dim60JoeKuoD6Init,
dim61JoeKuoD6Init,
dim62JoeKuoD6Init,
dim63JoeKuoD6Init,
dim64JoeKuoD6Init,
dim65JoeKuoD6Init,
dim66JoeKuoD6Init,
dim67JoeKuoD6Init,
dim68JoeKuoD6Init,
dim69JoeKuoD6Init,
dim70JoeKuoD6Init,
dim71JoeKuoD6Init,
dim72JoeKuoD6Init,
dim73JoeKuoD6Init,
dim74JoeKuoD6Init,
dim75JoeKuoD6Init,
dim76JoeKuoD6Init,
dim77JoeKuoD6Init,
dim78JoeKuoD6Init,
dim79JoeKuoD6Init,
dim80JoeKuoD6Init,
dim81JoeKuoD6Init,
dim82JoeKuoD6Init,
dim83JoeKuoD6Init,
dim84JoeKuoD6Init,
dim85JoeKuoD6Init,
dim86JoeKuoD6Init,
dim87JoeKuoD6Init,
dim88JoeKuoD6Init,
dim89JoeKuoD6Init,
dim90JoeKuoD6Init,
dim91JoeKuoD6Init,
dim92JoeKuoD6Init,
dim93JoeKuoD6Init,
dim94JoeKuoD6Init,
dim95JoeKuoD6Init,
dim96JoeKuoD6Init,
dim97JoeKuoD6Init,
dim98JoeKuoD6Init,
dim99JoeKuoD6Init,
dim100JoeKuoD6Init,
dim101JoeKuoD6Init,
dim102JoeKuoD6Init,
dim103JoeKuoD6Init,
dim104JoeKuoD6Init,
dim105JoeKuoD6Init,
dim106JoeKuoD6Init,
dim107JoeKuoD6Init,
dim108JoeKuoD6Init,
dim109JoeKuoD6Init,
dim110JoeKuoD6Init,
dim111JoeKuoD6Init,
dim112JoeKuoD6Init,
dim113JoeKuoD6Init,
dim114JoeKuoD6Init,
dim115JoeKuoD6Init,
dim116JoeKuoD6Init,
dim117JoeKuoD6Init,
dim118JoeKuoD6Init,
dim119JoeKuoD6Init,
dim120JoeKuoD6Init,
dim121JoeKuoD6Init,
dim122JoeKuoD6Init,
dim123JoeKuoD6Init,
dim124JoeKuoD6Init,
dim125JoeKuoD6Init,
dim126JoeKuoD6Init,
dim127JoeKuoD6Init,
dim128JoeKuoD6Init
};

template <int DIM>
class SobolRsg{

public:
  const static int W = 32;
  const static int bits_ = 8*sizeof(unsigned int);
  unsigned int directionIntegers_[DIM][W];
  unsigned int integerSequence_[DIM];
  unsigned int degree[DIM]; 
  unsigned int ppmt[DIM]; 
  bool firstDraw_;

  SobolRsg(){};

  void initialization(){
      // degree 0 is not used
      ppmt[0]=0;
      degree[0]=0;
      unsigned int k, index;
      unsigned int currentDegree=1;
      k=1;
      index=0;
      firstDraw_=true;

      unsigned int altDegree = maxAltDegree;

      for (; k<std::min<int>(DIM,altDegree); k++,index++)
      {
          ppmt[k] = AltPrimitivePolynomials[currentDegree-1][index];
          if (ppmt[k]==-1)
          {
              ++currentDegree;
              index=0;
              ppmt[k] = AltPrimitivePolynomials[currentDegree-1][index];
          }

          degree[k] = currentDegree;
      }

      // initializes bits_ direction integers for each dimension
      // and store them into directionIntegers_[dimensionality_][bits_]
      //
      // In each dimension k with its associated primitive polynomial,
      // the first degree_[k] direction integers can be chosen freely
      // provided that only the l leftmost bits can be non-zero, and
      // that the l-th leftmost bit must be set

      // degenerate (no free direction integers) first dimension
      int j;
      for (j=0; j<bits_; j++)
          directionIntegers_[0][j] = (1UL<<(bits_-j-1));


    for(int k=1;k<DIM;k++){
      int j=0;
      while(JoeKuoD6initializers[k-1][j] != 0UL){
        directionIntegers_[k][j]= JoeKuoD6initializers[k-1][j];
        directionIntegers_[k][j] <<= (bits_-j-1);
        j++;
      }
    }
      // computation of directionIntegers_[k][l] for l>=degree_[k]
      // by recurrence relation
      for (int k=1; k<DIM; k++) {
          unsigned int gk = degree[k];
          for (int l=gk; l<bits_; l++) {
              // eq. 8.19 "Monte Carlo Methods in Finance" by P. J�ckel
              unsigned int n = (directionIntegers_[k][l-gk]>>gk);
              // a[k][j] are the coefficients of the monomials in ppmt[k]
              // The highest order coefficient a[k][0] is not actually
              // used in the recurrence relation, and the lowest order
              // coefficient a[k][gk] is always set: this is the reason
              // why the highest and lowest coefficient of
              // the polynomial ppmt[k] are not included in its encoding,
              // provided that its degree is known.
              // That is: a[k][j] = ppmt[k] >> (gk-j-1)
              for (int j=1; j<gk; j++) {
                  // XORed with a selection of (unshifted) direction
                  // integers controlled by which of the a[k][j] are set
                  if ((ppmt[k] >> (gk-j-1)) & 1UL)
                      n ^= directionIntegers_[k][l-j];
              }
              // a[k][gk] is always set, so directionIntegers_[k][l-gk]
              // will always enter
              n ^= directionIntegers_[k][l-gk];
              directionIntegers_[k][l]=n;
          }
      }

      // initialize the Sobol integer/double vectors
      // first draw
      for (k=0; k<DIM; k++) {
          integerSequence_[k]=directionIntegers_[k][0];
      }
  }

  void nextInt32Sequence(unsigned int v[DIM]){
    if(firstDraw_){
      firstDraw_=false;
      return;
    }
    static unsigned int sequenceCounter_=0;
    sequenceCounter_++;
    unsigned int n=sequenceCounter_;
    int j=0;
    while(n&1){ n >>= 1; j++; }
    for(int k=0;k<DIM;k++){
      v[k] ^= directionIntegers_[k][j];
    }
  }

//const static double normalizationFactor_ =0.5/(1UL<<(SobolRsg::bits_-1));
  void nextSequence(double point[DIM]){
    nextInt32Sequence(integerSequence_);
    for(int k=0;k<DIM;k++){
      point[k]=(double)integerSequence_[k]*(0.5/((unsigned long)1<<31));//*normalizationFactor_;
    }

  }
};
}


#endif

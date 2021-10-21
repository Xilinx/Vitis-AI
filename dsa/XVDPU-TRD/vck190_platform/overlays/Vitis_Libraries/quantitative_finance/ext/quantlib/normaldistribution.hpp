/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2002, 2003 Ferdinando Ametrano
 Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl
 Copyright (C) 2010 Kakhkhor Abdijalilov

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

/*! \file normaldistribution.hpp
    \brief normal, cumulative and inverse cumulative distributions
*/

/* This file comes from open source QuantLib, https://github.com/lballabio/QuantLib.git. The license for it is BSD-3-Clause.
 * This file is used by XILINX as a reference.
 * In order to simplify its usage, some modification is added to the source file, such as change the data type, remove some not used codes.
 * The modification is done by XILINX.
 */

#ifndef quantlib_normal_distribution_hpp
#define quantlib_normal_distribution_hpp


namespace QuantLib {


#define QL_MAX_REAL            ((std::numeric_limits<double>::max)())
#define QL_MIN_REAL           -((std::numeric_limits<double>::max)())
#define QL_EPSILON             ((std::numeric_limits<double>::epsilon)())

      inline bool close_enough(double x, double y, int n) {
          // Deals with +infinity and -infinity representations etc.
          if (x == y)
              return true;
          double diff = std::fabs(x-y), tolerance = n * QL_EPSILON;
    
          if (x * y == 0.0) // x or y = 0.0
              return diff < (tolerance * tolerance);
    
          return diff <= tolerance*std::fabs(x) ||
                 diff <= tolerance*std::fabs(y);
      }
      
      inline bool close_enough(double x, double y) {
          return close_enough(x,y,42);
      }
    //! Inverse cumulative normal distribution function
    /*! Given x between zero and one as
      the integral value of a gaussian normal distribution
      this class provides the value y such that
      formula here ...

      It use Acklam's approximation:
      by Peter J. Acklam, University of Oslo, Statistics Division.
      URL: http://home.online.no/~pjacklam/notes/invnorm/index.html

      This class can also be used to generate a gaussian normal
      distribution from a uniform distribution.
      This is especially useful when a gaussian normal distribution
      is generated from a low discrepancy uniform distribution:
      in this case the traditional Box-Muller approach and its
      variants would not preserve the sequence's low-discrepancy.

    */
    template<typename Real>
    class InverseCumulativeNormal
        : public std::unary_function<Real,Real> {
      public:
        InverseCumulativeNormal(Real average = 0.0,
                                Real sigma   = 1.0){
            average_ = average;
            sigma_ = sigma;
        }
        // function
        Real operator()(Real x) {
            return average_ + sigma_*standard_value(x);
        }
        // value for average=0, sigma=1
        /* Compared to operator(), this method avoids 2 floating point
           operations (we use average=0 and sigma=1 most of the
           time). The speed difference is noticeable.
        */
        Real standard_value(Real x) {
            Real z;
            if (x < x_low_ || x_high_ < x) {
                z = tail_value(x);
            } else {
                z = x - 0.5;
                Real r = z*z;
                z = (((((a1_*r+a2_)*r+a3_)*r+a4_)*r+a5_)*r+a6_)*z /
                    (((((b1_*r+b2_)*r+b3_)*r+b4_)*r+b5_)*r+1.0);
            }
            //// The relative error of the approximation has absolute value less
            //// than 1.15e-9.  One iteration of Halley's rational method (third
            //// order) gives full machine precision.
            //// #define REFINE_TO_FULL_MACHINE_PRECISION_USING_HALLEYS_METHOD
            //#ifdef REFINE_TO_FULL_MACHINE_PRECISION_USING_HALLEYS_METHOD
            //// error (f_(z) - x) divided by the cumulative's derivative
            //const Real r = (f_(z) - x) * M_SQRT2 * M_SQRTPI * exp(0.5 * z*z);
            ////  Halley's method
            //z -= r/(1+0.5*z*r);
            //#endif

            return z;
        }
      private:
        /* Handling tails moved into a separate method, which should
           make the inlining of operator() and standard_value method
           easier. tail_value is called rarely and doesn't need to be
           inlined.
        */
        //static Real tail_value(Real x);
        //#if defined(QL_PATCH_SOLARIS)
        //CumulativeNormalDistribution f_;
        //#else
        //static const CumulativeNormalDistribution f_;
        //#endif
        Real average_, sigma_;

        // Coefficients for the rational approximation.
        const Real a1_ = -3.969683028665376e+01;
        const Real a2_ =  2.209460984245205e+02;
        const Real a3_ = -2.759285104469687e+02;
        const Real a4_ =  1.383577518672690e+02;
        const Real a5_ = -3.066479806614716e+01;
        const Real a6_ =  2.506628277459239e+00;

        const Real b1_ = -5.447609879822406e+01;
        const Real b2_ =  1.615858368580409e+02;
        const Real b3_ = -1.556989798598866e+02;
        const Real b4_ =  6.680131188771972e+01;
        const Real b5_ = -1.328068155288572e+01;

        const Real c1_ = -7.784894002430293e-03;
        const Real c2_ = -3.223964580411365e-01;
        const Real c3_ = -2.400758277161838e+00;
        const Real c4_ = -2.549732539343734e+00;
        const Real c5_ =  4.374664141464968e+00;
        const Real c6_ =  2.938163982698783e+00;

        const Real d1_ =  7.784695709041462e-03;
        const Real d2_ =  3.224671290700398e-01;
        const Real d3_ =  2.445134137142996e+00;
        const Real d4_ =  3.754408661907416e+00;

        // Limits of the approximation regions
        const Real x_low_ = 0.02425;
        const Real x_high_= 1.0 - x_low_;

        Real tail_value(Real x) {
            if (x <= 0.0 || x >= 1.0) {
                // try to recover if due to numerical error
                if (close_enough(x, 1.0)) {
                    return QL_MAX_REAL; // largest value available
                } else if (std::fabs(x) < QL_EPSILON) {
                    return QL_MIN_REAL; // largest negative value available
                } else {
                    std::cout<<"InverseCumulativeNormal(" << x
                             << ") undefined: must be 0 < x < 1"<<std::endl;
                }
            }

            Real z;
            if (x < x_low_) {
                // Rational approximation for the lower region 0<x<u_low
                z = std::sqrt(-2.0*std::log(x));
                z = (((((c1_*z+c2_)*z+c3_)*z+c4_)*z+c5_)*z+c6_) /
                    ((((d1_*z+d2_)*z+d3_)*z+d4_)*z+1.0);
            } else {
                // Rational approximation for the upper region u_high<x<1
                z = std::sqrt(-2.0*std::log(1.0-x));
                z = -(((((c1_*z+c2_)*z+c3_)*z+c4_)*z+c5_)*z+c6_) /
                    ((((d1_*z+d2_)*z+d3_)*z+d4_)*z+1.0);
            }

            return z;
       }
};
}//namespace

#endif

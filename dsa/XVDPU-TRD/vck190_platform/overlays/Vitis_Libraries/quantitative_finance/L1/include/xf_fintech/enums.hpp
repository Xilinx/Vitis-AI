/*
 * Copyright 2019 Xilinx, Inc.
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
/**
 * @file enums.hpp
 * @brief Header of Common enums
 */
#ifndef XF_FINTECH_ENUMS_H
#define XF_FINTECH_ENUMS_H

namespace xf {
namespace fintech {

// Interest rate compounding rule
enum Compounding {
    Simple = 0,           /* 1 + rt */
    Compounded = 1,       /* (1 + r)^t */
    Continuous = 2,       /* e^rt */
    SimpleThenCompounded, /* Simple up to the first period then compounded. */
    CompoundedThenSimple  /* Compounded up to the first period then simple. */
};

// frequency of events
enum Frequency {
    NoFrequency = -1,     /* null frequency */
    Once = 0,             /* only once */
    Annual = 1,           /* once a year */
    Semiannual = 2,       /* twice a year */
    EveryFourthMonth = 3, //!< every fourth month
    Quarterly = 4,        //!< every third month
    Bimonthly = 6,        //!< every second month
    Monthly = 12,         //!< once a month
    EveryFourthWeek = 13, //!< every fourth week
    Biweekly = 26,        //!< every second week
    Weekly = 52,          //!< once a week
    Daily = 365,          //!< once a day
    OtherFrequency = 999  //!< some other unknown frequency
};

// option type put or call
enum Type { Put = -1, Call = 1 };

// Business Day conventions
// These conventions specify the algorithm used to adjust a date in case
//        it is not a valid business day.
enum BusinessDayConvention {
    // ISDA
    Following,         /* Choose the first business day after the given holiday.*/
    ModifiedFollowing, /* Choose the first business day after the given holiday unless it belongs
                          to a different month, in which case choose the first business day before the holiday.*/
    Preceding,         /* Choose the first business day before the given holiday. */
    // NON ISDA
    ModifiedPreceding,          /* Choose the first business day before the given holiday unless it belongs
                                   to a different month, in which case choose the first business day after
                                   the holiday. */
    Unadjusted,                 /* Do not adjust. */
    HalfMonthModifiedFollowing, /* Choose the first business day after the given holiday unless that day
                                   crosses the mid-month (15th) or the end of month, in which case choose
                                   the first business day before the holiday. */
    Nearest                     /* Choose the nearest business day to the given holiday. If both the
                                   preceding and following business days are equally far away, default
                                   to following business day. */
};

namespace enums {
/**
 * @brief Discrete type
 */
enum DiscreType {
    kDTPartialTruncation,
    kDTFullTruncation,
    kDTReflection,
    kDTQuadraticExponential,
    kDTQuadraticExponentialMartingale
};
/**
 * @brief Model type
 */
enum ModelType { BS, Heston };
/**
 * @brief Random number generator type
 */
enum RNGType { PseudoRandom, LowDiscrepancy };
/**
 * @brief Option Style
 */
enum OptionStyle {
    LongstaffSchwartz,
    American,
    European,
    Digital,
    BarrierBiased,
    BarrierNoBiased,
    Cliquet,
    Asian_AP,
    Asian_AS,
    Asian_GP,
    EuropeanBypass
};
/**
 * @brief Barrier Option type
 */
enum BarrierType { DownIn, DownOut, UpIn, UpOut };
} // namespace enums
using namespace enums;
} // namespace fintech
} // namespace xf

#endif // define XF_FINTECH_ENUMS_H

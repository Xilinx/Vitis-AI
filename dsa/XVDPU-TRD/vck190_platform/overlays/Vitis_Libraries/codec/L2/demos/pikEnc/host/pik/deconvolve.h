// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DECONVOLVE_H_
#define PIK_DECONVOLVE_H_

namespace pik {

// Compute a filter such that convolving with it is an approximation of the
// inverse of convolving with the provided filter.
// The resulting filter is written into inverse_filter and is of the provided
// inverse_filter_length length.
// filter_length and inverse_filter_length have to be odd.
// Returns the L2 distance between the identity filter and the composition of
// the two filters.
float InvertConvolution(const float* filter, int filter_length,
                        float* inverse_filter, int inverse_filter_length);

}  // namespace pik

#endif  // PIK_DECONVOLVE_H_

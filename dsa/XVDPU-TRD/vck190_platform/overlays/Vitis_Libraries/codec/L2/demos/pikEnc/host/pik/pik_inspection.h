// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.


#ifndef PIK_PIK_INSPECTION_H_
#define PIK_PIK_INSPECTION_H_

#include <functional>
#include "pik/image.h"

namespace pik {
// Type of the inspection-callback which, if enabled, will be called on various
// intermediate data during image processing, allowing inspection access.
//
// Returns false if processing can be stopped at that point, true otherwise.
// This is only advisory - it is always OK to just continue processing.
using InspectorImage3F = std::function<bool(const char*, const Image3F&)>;
}  // namespace pik

#endif  // PIK_PIK_INSPECTION_H_

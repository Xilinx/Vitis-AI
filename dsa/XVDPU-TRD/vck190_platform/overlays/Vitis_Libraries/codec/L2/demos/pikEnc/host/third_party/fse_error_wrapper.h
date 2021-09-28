// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// fse_decompress.c uses a "#define FSE_isError" so we can't use that define in
// our wrapper for that specific file. This removes the #define FSE_isError from
// our wrapper, which is meant to be included only on that file.

#ifndef __THIRD_PARTY_FINISTESTATEENTROPY_FSE_ERROR_WRAPPER_H__
#define __THIRD_PARTY_FINISTESTATEENTROPY_FSE_ERROR_WRAPPER_H__

#undef FSE_isError

#endif // __THIRD_PARTY_FINISTESTATEENTROPY_FSE_ERROR_WRAPPER_H__

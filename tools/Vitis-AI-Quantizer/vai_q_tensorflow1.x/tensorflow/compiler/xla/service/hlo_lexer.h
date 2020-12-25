/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LEXER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LEXER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Defines different kinds of tokens used by the HLO lexer.
//
// You shouldn't need to use this directly unless you're using HloLexer
// directly, and you probably don't need to do that.  Use hlo_parser instead.
enum class TokKind {
  // Markers
  kEof,
  kError,

  // Tokens with no info.
  kEqual,     // =
  kComma,     // ,
  kColon,     // :
  kAsterisk,  // *
  kLsquare,
  kRsquare,  // [  ]
  kLbrace,
  kRbrace,  // {  }
  kLparen,
  kRparen,  // (  )
  kDots,    // ...

  kArrow,  // ->
  kLeq,    // <=

  // Keywords
  kw_HloModule,
  kw_ENTRY,
  kw_ROOT,
  kw_true,
  kw_false,
  kw_maximal,
  kw_replicated,
  kw_nan,
  kw_inf,
  kw_sparse,

  kNegInf,  // -inf

  // Typed tokens.
  kPrimitiveType,  // F32, PRED, etc.
  kName,           // %foo
  kAttributeName,  // dimensions=
  kDimLabels,      // [0-9bf]{2,}_[0-9io]{2,}->[0-9bf]{2,}
  kDxD,            // [0-9]+(x[0-9]+)+
  kPad,            // [0-9]+_[0-9]+(_[0-9]+)?(x[0-9]+_[0-9]+(_[0-9]+)?)*
  kIdent,          // other identifiers
  kString,         // "abcd\"\n"
  kInt,            // 42
  kDecimal,        // 4.2
};

string TokKindToString(TokKind kind);

// Lexer for the HloModule::ToString() format text.
//
// This class is meant to be used by hlo_parser.cc.  You shouldn't need to use
// it directly.
class HloLexer {
 public:
  explicit HloLexer(absl::string_view buf) : buf_(buf) {
    current_ptr_ = buf_.begin();
  }

  TokKind Lex() { return token_state_.current_kind = LexToken(); }

  TokKind GetKind() const { return token_state_.current_kind; }
  string GetStrVal() const {
    switch (GetKind()) {
      case TokKind::kName:
      case TokKind::kAttributeName:
      case TokKind::kDimLabels:
      case TokKind::kDxD:
      case TokKind::kPad:
      case TokKind::kString:
      case TokKind::kIdent:
        return token_state_.str_val;
      default:
        LOG(FATAL) << "This token does not have string value";
    }
  }
  int64 GetInt64Val() const {
    CHECK(GetKind() == TokKind::kInt);
    return token_state_.int64_val;
  }
  double GetDecimalVal() const {
    CHECK(GetKind() == TokKind::kDecimal);
    return token_state_.decimal_val;
  }
  PrimitiveType GetPrimitiveTypeVal() const {
    CHECK(GetKind() == TokKind::kPrimitiveType);
    return token_state_.primitive_type_val;
  }

  typedef const char* LocTy;

  // Returns the location of the current token.
  LocTy GetLoc() const { return token_state_.token_start; }

  // Returns the line and column of a location in the buffer.
  std::pair<unsigned, unsigned> GetLineAndColumn(LocTy location) const;

  // Returns the whole line given the location.
  absl::string_view GetLine(LocTy loc) const;

  // Looks ahead one token and returns it. Lexer state is unchanged.
  TokKind LookAhead();

 private:
  // Returns the current character. If it's neither the end of input buffer nor
  // an invalid character, moves the pointer forward.
  int GetNextChar();

  // Returns the current character.
  int PeekCurrentChar() const;

  // Creates StringPiece with the given begin and end. Exits if the begin > end,
  // or it's out of the range of the current buffer.
  absl::string_view StringPieceFromPointers(const char* begin,
                                            const char* end) const;

  // Returns true if the given ptr is dereferenceable within the range of the
  // current buffer.
  bool CanDereference(const char* ptr) const;

  TokKind LexToken();

  TokKind LexIdentifier();
  TokKind LexPercent();
  TokKind LexShape();
  TokKind LexConstant();
  TokKind LexNumberOrPattern();
  TokKind LexString();

  const absl::string_view buf_;
  const char* current_ptr_;

  // Information about the current token.
  struct TokenState {
    const char* token_start = nullptr;
    TokKind current_kind;
    string str_val;
    int64 int64_val;
    double decimal_val;
    PrimitiveType primitive_type_val;
  };
  TokenState token_state_;

  struct LineNoCacheTy {
    const char* last_query;
    unsigned line_no_of_query;
  };
  // This caches the line number of the previous query.
  mutable LineNoCacheTy line_no_cache_{nullptr, 0};
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LEXER_H_

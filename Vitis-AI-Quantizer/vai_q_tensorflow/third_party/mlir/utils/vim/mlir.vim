" Copyright 2019 The MLIR Authors.
"
" Licensed under the Apache License, Version 2.0 (the "License");
" you may not use this file except in compliance with the License.
" You may obtain a copy of the License at
"
"     http://www.apache.org/licenses/LICENSE-2.0
"
" Unless required by applicable law or agreed to in writing, software
" distributed under the License is distributed on an "AS IS" BASIS,
" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
" See the License for the specific language governing permissions and
" limitations under the License.

" Vim syntax file
" Language: MLIR

" quit when a syntax file was already loaded
if exists("b:current_syntax")
  finish
endif

syn keyword mlirType index i1 i2 i4 i8 i13 i16 i32 i64
      \ f16 f32 tf_control
syn keyword mlirType memref tensor vector

syntax keyword mlirKeywords extfunc cfgfunc mlfunc for to step return
syntax keyword mlirConditional if else
syntax keyword mlirCoreOps dim addf addi subf subi mulf muli cmpi select constant affine.apply call call_indirect extract_element getTensor memref_cast tensor_cast load store alloc dealloc dma_start dma_wait

syn match mlirInt "-\=\<\d\+\>"
syn match mlirFloat "-\=\<\d\+\.\d\+\>"
syn match mlirMapOutline "#.*$"
syn match mlirOperator      "[+\-*=]"

syn region mlirComment start="//" skip="\\$" end="$"
syn region mlirString matchgroup=mlirString start=+"+ end=+"+

hi def link mlirComment      Comment
hi def link mlirKeywords     Instruction
hi def link mlirCoreOps      Instruction
hi def link mlirInt          Constant
hi def link mlirType         Type
hi def link mlirMapOutline   PreProc
hi def link mlirConditional  Conditional
hi def link mlirString       String
hi def link mlirOperator     Operator
hi def link mlirInstruction  Operator
hi def link mlirAffineOp     Operator

let b:current_syntax = "mlir"

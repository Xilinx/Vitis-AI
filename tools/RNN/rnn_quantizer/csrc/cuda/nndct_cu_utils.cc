

/*
* Copyright 2019 Xilinx Inc.
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


#include "../../include/cuda/nndct_cu_utils.h"
dim3 GetGridSizeF(unsigned n){
  dim3 Gr;
  unsigned nb=( n + BLOCKSIZE - 1 ) / BLOCKSIZE;
  if(nb<65536){
      Gr.x=nb;
      Gr.y=1;
  }else{
      float tmp=nb;
      float sqrt_val=sqrt(tmp);
      unsigned x=sqrt_val;
      Gr.x=x;
      unsigned y=(nb+Gr.x-1)/Gr.x;
      Gr.y =y;
  }
  Gr.z = 1;
  return Gr;
}

void GetBlockSizesForSimpleMatrixOperation(int num_rows,
                                           int num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock) {
  int col_blocksize = BLOCKSIZE_COL, row_blocksize = BLOCKSIZE_ROW;
  while (col_blocksize > 1 &&
         (num_cols + (num_cols / 2) <= col_blocksize ||
          num_rows > 65536 * row_blocksize)) {
    col_blocksize /= 2;
    row_blocksize *= 2;
  }

  dimBlock->x = col_blocksize;
  dimBlock->y = row_blocksize;
  dimBlock->z = 1;
  dimGrid->x = n_blocks(num_cols, col_blocksize);
  dimGrid->y = n_blocks(num_rows, row_blocksize);
  dimGrid->z = 1;
}




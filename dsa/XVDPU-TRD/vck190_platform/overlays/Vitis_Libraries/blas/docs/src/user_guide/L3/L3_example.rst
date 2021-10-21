.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: BLAS, Library, Vitis BLAS Library, Vitis BLAS, example, level 3
   :description: Vitis BLAS library level 3 appliction programming interface example.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _example_l3:

=====================
L3 API example
=====================
For example code references please follow the link below. 

**1. Vitis BLAS L3 compilation**

All examples provided here could be built with compilation steps similar to the following, target could be either hw or hw_emu(for testing hw emulation)

.. code-block:: bash

  make host TARGET=hw
  
**2. Vitis BLAS L3 run**

Examples could be run with the following steps, target could be either hw or hw_emu(for testing hw emulation)

.. code-block:: bash

  make run TARGET=hw PLATFORM_REPO_PATHS=LOCAL_PLATFORM_PATH


**3. Vitis BLAS L3 code example**

The following is an example of how to use Vitis BLAS API. Users always need to include header file xf_blas.hpp. 

.. code-block:: c++

  #include "xf_blas.hpp"
  
  # define IDX2R(i,j,ld) (((i)*( ld ))+(j))
  # define m 128 // a - mxk matrix
  # define n 128 // b - kxn matrix
  # define k 128 // c - mxn matrix
  
  using namespace std;
  
  int main(int argc, char **argv) {
    
    if (argc < 3){
      cerr << " usage: \n"
           << " gemx_example.exe gemx.xclbin config_info.dat\n";
      return EXIT_FAILURE; 
    }
    unsigned int l_argIdx = 1;
    string l_xclbinFile(argv[l_argIdx++]);
    string l_configFile(argv[l_argIdx++]);  
    
    int i, j; // i-row index ,j- column index
    float * a, * b, * c;
    a = ( float *) malloc (m*k* sizeof ( float )); 
    b = ( float *) malloc (k*n* sizeof ( float )); 
    c = ( float *) malloc (m*n* sizeof ( float )); 
  
    int ind = 1;
    for( i = 0; i<  m; i ++){ 
      for( j = 0; j < k; j ++){ 
        a[ IDX2R (i,j,k )]= (float) ind++; 
      } 
    } 
    ind = 1;
    for( i = 0; i<  k; i ++){ 
      for( j = 0; j < n; j ++){ 
        b[ IDX2R (i,j,n )]= (float) ind++; 
      } 
    } 
  
    for( i = 0; i<  m; i ++){ 
      for( j = 0; j < n; j ++){ 
        c[ IDX2R (i,j,n )]= 0; 
      } 
    } 
  
    xfblasEngine_t engineName = XFBLAS_ENGINE_GEMM;
    xfblasStatus_t status = xfblasCreate(l_xclbinFile.c_str(), l_configFile, engineName);
    status = xfblasMallocRestricted(m,k,sizeof(*a),a,k);
    status = xfblasMallocRestricted(k,n,sizeof(*b),b,n);
    status = xfblasMallocRestricted(m,n,sizeof(*c),c,n);
    status = xfblasSetMatrixRestricted(a);
    status = xfblasSetMatrixRestricted(b);
    status = xfblasSetMatrixRestricted(c);
    status = xfblasGemm(XFBLAS_OP_N, XFBLAS_OP_N, m, n, k, 1, a, k, b, n, 1, c, n);
    status = xfblasGetMatrixRestricted(c);
    
    for ( i = 0; i < 128; i ++){
      for ( j = 0; j < 128; j ++){
        cout<< (c[ IDX2R (i,j, k )])<<" ";
      }
      cout<<"\n";
    }
    
    
    xfblasFree(a);
    xfblasFree(b);
    xfblasFree(c);
    xfblasDestroy();
    free(a);
    free(b);
    free(c);
    
    return EXIT_SUCCESS;
  }


These API functions run on the first kernel by default, but they could support multi-kernels xclbin. Users will need to put numbers of kernels or index of the kernel in those functions to let the API know. The following code shows how to initialize Vitis BLAS library to support a xclbin with 2 kernels and run with the second kernel.

.. code-block:: c++
   
    xfblasStatus_t status = xfblasCreate(l_xclbinFile.c_str(), l_configFile, XFBLAS_ENGINE_GEMM, 2);
    status = xfblasMallocRestricted(m,k,sizeof(*a),a,k, 1);
    status = xfblasMallocRestricted(k,n,sizeof(*b),b,n, 1);
    status = xfblasMallocRestricted(m,n,sizeof(*c),c,n, 1);
    status = xfblasSetMatrixRestricted(a, 1);
    status = xfblasSetMatrixRestricted(b, 1);
    status = xfblasSetMatrixRestricted(c, 1);
    status = xfblasGemm(XFBLAS_OP_N, XFBLAS_OP_N, m, k, n, 1, a, k, b, n, 1, c, n, 1);
    status = xfblasGetMatrixRestricted(c, 1);
  
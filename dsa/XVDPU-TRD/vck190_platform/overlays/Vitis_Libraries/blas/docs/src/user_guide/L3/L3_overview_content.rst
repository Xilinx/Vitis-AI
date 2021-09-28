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
   :keywords: BLAS, Library, Vitis BLAS Library, Vitis BLAS, level 3
   :description: Vitis BLAS library level 3 provides software API functions to offload BLAS operations to pre-built FPGA images.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _user_guide_overview_content_l3:



Vitis BLAS level 3 provides software API functions to offload BLAS operations to pre-built FPGA images. 

1. Introduction
================
The Vitis BLAS level 3 library is an implementation of BLAS on top of the XILINX runtime (XRT). It allows software developers to use Vitis BLAS library without writing any runtime functions and hardware configurations. 

1.1 Data layout
---------------
Vitis BLAS library uses row-major storage. The array index of a matrix element could be calculated by the following macro.
  
.. code-block::

  # define IDX2R(i,j,ld) ( ( (i)*(ld) ) + (j) )
  
1.2 Memory Allocation
----------------------
Vitis BLAS level 3 library supports three different versions of APIs to support memory allocation in device. Users could choose from different versions to better support their application based on the cases. Examples for using three different versions could be found in :doc:`BLAS Level 3 example<L3_example>`.

+--------------------------------+-------------------------------------+------------------------------+
| Already have host memory input | Host memory matrix sizes are padded | Version to choose            |
+================================+=====================================+==============================+
| Yes                            | Yes                                 | Restricted memory version    |
+--------------------------------+-------------------------------------+------------------------------+
| Yes                            | No                                  | Default memory version       |
+--------------------------------+-------------------------------------+------------------------------+
| No                             | Does not apply                      | Pre-allocated memory version |
+--------------------------------+-------------------------------------+------------------------------+ 

Restricted memory version
^^^^^^^^^^^^^^^^^^^^^^^^^^
  To use restricted memory version, user's input matrix sizes must be multiple of certain configuration values that are used to build the FPGA bitstreams. Also, host memory is encouraged to be 4k aligned when using restricted memory version. Compared to the default memory version, even though there are requirements on the matrix sizes, restricted memory version could save extra memory copy in host side. 

Default memory version
^^^^^^^^^^^^^^^^^^^^^^^
  This version has no limitations on user host memory, and it is easy to use. API functions will do the padding internally so this will lead to extra memory copy in host side. The result output matrix will also be the same sizes.
  
Pre-allocated memory version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  To use this version, users need to call API functions to allocate the device memory first, then fill in host memory that is mapped to device memory with values. There is no extra memory copy and the programming is easier compared to the other two versions. However, when filling in the matrices, users need to use the padded sizes, also the result output matrix's sizes are padded instead of the original ones. Please see examples for more usage information. 
  
  
1.3 Supported Datatypes
------------------------
- float

2. Using the Vitis BLAS API
=============================

2.1 General description
------------------------
This section describes how to use the Vitis BLAS library API level.

2.1.1 Error status
^^^^^^^^^^^^^^^^^^^
Vitis BLAS API function calls return the error status of datatype `xfblasStatus_t <2.2.1 xfblasStatus_t_>`_.

2.1.2 Vitis BLAS initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To initialize the library, xfblasCreate() function must be called. This function will open the device, download the FPGA image to the device  and create context on the selected compute unit. For a multi-kernels xclbin, contexts will be opened on the corresponding compute units. Please refer L3 examples for detail usage.

2.2 Datatypes Reference
-----------------------

2.2.1 xfblasStatus_t
^^^^^^^^^^^^^^^^^^^^^^
The type is used for function status returns. All Vitis BLAS level 3 library functions return status which has the following values.

+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+--------+
| Item                          | Meaning                                                                                                               | Value  |
+===============================+=======================================================================================================================+========+
| XFBLAS_STATUS_SUCCESS         | The function is completed successfully                                                                                | 0      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+--------+
| XFBLAS_STATUS_NOT_INITIALIZED | The Vitis BLAS library was not initialized. This is usually caused by not calling function xfblasCreate previously.   | 1      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+--------+
| XFBLAS_STATUS_INVALID_VALUE   | An unsupported value or parameter was passed to the function. For example, an negative matrix size.                   | 2      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+--------+
| XFBLAS_STATUS_ALLOC_FAILED    | Memory allocation failed inside the Vitis BLAS library.                                                               | 3      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+--------+
| XFBLAS_STATUS_NOT_SUPPORTED   | The functionality requested is not supported yet.                                                                     | 4      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+--------+
| XFBLAS_STATUS_NOT_PADDED      | For restricted mode, matrix sizes are not padded correctly.                                                           | 5      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+--------+

2.2.2 xfblasEngine_t
^^^^^^^^^^^^^^^^^^^^^
The xfblasEngine_t type indicates which engine needs to be performed when initializes the Vitis BLAS library. xfblasEngine_t type should be matched with the FPGA bitstream.

+--------------------+-----------------------------+
| Value              | Meaning                     |
+====================+=============================+
| XFBLAS_ENGINE_GEMM | The GEMM engine is selected |
+--------------------+-----------------------------+


2.2.3 xfblasOperation_t
^^^^^^^^^^^^^^^^^^^^^^^^
The xfblasOperation_t type indicates which operation needs to be performed with the matrix.

+-------------+-----------------------------------------------+
| Value       | Meaning                                       |
+=============+===============================================+
| XFBLAS_OP_N | The non-transpose operation is selected       |
+-------------+-----------------------------------------------+
| XFBLAS_OP_T | The transpose operation is selected           |
+-------------+-----------------------------------------------+
| XFBLAS_OP_C | The conjugate transpose operation is selected |
+-------------+-----------------------------------------------+

2.3 Vitis BLAS Helper Function Reference
----------------------------------------------

2.3.1 xfblasCreate
^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasCreate(const char* xclbin, string configFile, const char* logFile, xfblasEngine_t engineName, unsigned int kernelNumber = 1, unsigned int deviceIndex = 0)

This function initializes the Vitis BLAS library and creates a handle for the specific engine. It must be called prior to any other Vitis BLAS library calls.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - xclbin
        - file path to FPGA bitstream
    *
        - configFile
        - file path to config_info.dat file
    *
        - logFile
        - file path to log file
    *
        - engineName
        - Vitis BLAS engine to run
    *
        - kernelNumber
        - number of kernels that is being used, default is 1
    *
        - deviceIndex
        - index of device that is being used, default is 0

.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the initialization succeeded
    *
        - xfblasStatus_t
        - 1 if the opencl runtime initialization failed
    *
        - xfblasStatus_t
        - 2 if the xclbin doesn't contain the engine
    *
        - xfblasStatus_t
        - 4 if the engine is not supported for now

2.3.2 xfblasFree
^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasFree(void* A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function frees memory in FPGA device.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - A
        - pointer to matrix A in the host memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0


.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix
        
2.3.3 xfblasDestroy
^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasDestroy(unsigned int kernelNumber = 1, unsigned int deviceIndex = 0)

This function releases handle used by the Vitis BLAS library.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - kernelNumber
        - number of kernels that is being used, default is 1
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80

    *
        - xfblasStatus_t
        - 0 if the shut down succeeded
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
        
2.3.4 xfblasMalloc
^^^^^^^^^^^^^^^^^^^
        
.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasMalloc(short** devPtr, int rows, int lda, int elemSize, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)
    xfblasStatus_t xfblasMalloc(float** devPtr, int rows, int lda, int elemSize, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function allocates memory on the FPGA device.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - devPtr
        - pointer to mapped memory
    *
        - rows
        - number of rows in the matrix
    *
        - lda
        - leading dimension of the matrix that indicates the total number of cols in the matrix
    *
        - elemSize
        - number of bytes required to store each element in the matrix
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the allocation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 2 if parameters rows, cols, elemSize, lda <= 0 or cols > lda or data types are not matched
    *
        - xfblasStatus_t
        - 3 if there is memory already allocated to the same matrix
    *
        - xfblasStatus_t
        - 4 if the engine is not supported for now

2.3.5 xfblasSetVector
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasSetVector(int n, int elemSize, short* x, int incx, short* d_x, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)
    xfblasStatus_t xfblasSetVector(int n, int elemSize, float* x, int incx, float* d_x, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function copies a vector in host memory to FPGA device memory. `xfblasMalloc() <2.3.4 xfblasMalloc_>`_ need to be called prior to this function.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - n
        - number of elements in vector
    *
        - elemSize
        - number of bytes required to store each element in the vector
    *
        - x
        - pointer to the vector in the host memory
    *
        - incx
        - the storage spacing between consecutive elements of vector x
    *
        - d_x
        - pointer to mapped memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80

    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 2 if parameters rows, cols, elemSize, lda <= 0 or cols > lda or data types are not matched
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the vector
    *
        - xfblasStatus_t
        - 4 if the engine is not supported for now

2.3.6 xfblasGetVector
^^^^^^^^^^^^^^^^^^^^^^
        
.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasGetVector(int n, int elemSize, short* d_x, short* x, int incx, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)
    xfblasStatus_t xfblasGetVector(int n, int elemSize, float* d_x, float* x, int incx, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function copies a vector in FPGA device memory to host memory.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - n
        - number of elements in vector
    *
        - elemSize
        - number of bytes required to store each element in the vector
    *
        - d_x
        - pointer to mapped memory
    *
        - x
        - pointer to the vector in the host memory
    *
        - incx
        - the storage spacing between consecutive elements of vector x
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80

    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the vector

2.3.7 xfblasSetMatrix
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasSetMatrix(int rows, int cols, int elemSize, short* A, int lda, short* d_A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)
    xfblasStatus_t xfblasSetMatrix(int rows, int cols, int elemSize, float* A, int lda, float* d_A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function copies a matrix in host memory to FPGA device memory. `xfblasMalloc() <2.3.4 xfblasMalloc_>`_ need to be called prior to this function.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - rows
        - number of rows in the matrix
    *
        - cols
        - number of cols in the matrix that is being used
    *
        - elemSize
        - number of bytes required to store each element in the matrix
    *
        - A
        - pointer to the matrix array in the host memory
    *
        - lda
        - leading dimension of the matrix that indicates the total number of cols in the matrix
    *
        - d_A
        - pointer to mapped memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 2 if parameters rows, cols, elemSize, lda <= 0 or cols > lda or data types are not matched
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix
    *
        - xfblasStatus_t
        - 4 if the engine is not supported for now

2.3.8 xfblasGetMatrix
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasGetMatrix(int rows, int cols, int elemSize, short* d_A, short* A, int lda, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)
    xfblasStatus_t xfblasGetMatrix(int rows, int cols, int elemSize, float* d_A, float* A, int lda, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) 

This function copies a matrix in FPGA device memory to host memory.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - rows
        - number of rows in the matrix
    *
        - cols
        - number of cols in the matrix that is being used

    *
        - elemSize
        - number of bytes required to store each element in the matrix
    *
        - d_A
        - pointer to mapped memory
    *
        - A
        - pointer to the matrix array in the host memory
    *
        - lda
        - leading dimension of the matrix that indicates the total number of cols in the matrix
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix
        
2.3.9 xfblasMallocRestricted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasMallocRestricted(int rows, int cols, int elemSize, void* A, int lda, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function allocates memory for host row-major format matrix on the FPGA device.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - rows
        - number of rows in the matrix
    *
        - cols
        - number of cols in the matrix that is being used
    *
        - elemSize
        - number of bytes required to store each element in the matrix
    *
        - A
        - pointer to the matrix array in the host memory
    *
        - lda
        - leading dimension of the matrix that indicates the total number of cols in the matrix
        
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the allocation completed successfully

    *
        - xfblasStatus_t
        - 1 if the library was not initialized

    *
        - xfblasStatus_t
        - 2 if parameters rows, cols, elemSize, lda <= 0 or cols > lda or data types are not matched

    *
        - xfblasStatus_t
        - 3 if there is memory already allocated to the same matrix

    *
        - xfblasStatus_t
        - 4 if the engine is not supported for now

    *
        - xfblasStatus_t
        - 5 if rows, cols or lda is not padded correctly

2.3.10 xfblasSetVectorRestricted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasSetVectorRestricted(void* x, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function copies a vector in host memory to FPGA device memory. `xfblasMallocRestricted() <2.3.9 xfblasMallocRestricted_>`_ need to be called prior to this function.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - x
        - pointer to the vector in the host memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the vector
  
2.3.11 xfblasGetVectorRestricted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasGetVectorRestricted(void* x, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function copies a matrix in FPGA device memory to host memory.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - x
        - pointer to vetcor x in the host memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80

    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix


2.3.12 xfblasSetMatrixRestricted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasSetMatrixRestricted(void* A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function copies a matrix in host memory to FPGA device memory. `xfblasMallocRestricted() <2.3.9 xfblasMallocRestricted_>`_ need to be called prior to this function.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - A
        - pointer to the matrix array in the host memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix

2.3.13 xfblasGetMatrixRestricted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasGetMatrixRestricted(void* A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function copies a matrix in FPGA device memory to host memory.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - A
        - pointer to matrix A in the host memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix

2.3.14 xfblasMallocManaged
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasMallocManaged(short** devPtr, int* paddedLda, int rows, int lda, int elemSize, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)
    xfblasStatus_t xfblasMallocManaged(float** devPtr, int* paddedLda, int rows, int lda, int elemSize, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function allocates memory on the FPGA device, rewrites the leading dimension size after padding.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - devPtr
        - pointer to mapped memory
    *
        - paddedLda
        - leading dimension of the matrix after padding
    *
        - rows
        - number of rows in the matrix
    *
        - lda
        - leading dimension of the matrix that indicates the total number of cols in the matrix
    *
        - elemSize
        - number of bytes required to store each element in the matrix       
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80        

    *
        - xfblasStatus_t
        - 0 if the allocation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 2 if parameters rows, cols, elemSize, lda <= 0 or cols > lda or data types are not matched
    *
        - xfblasStatus_t
        - 3 if there is memory already allocated to the same matrix
    *
        - xfblasStatus_t
        - 4 if the engine is not supported for now
        
        
2.3.15 xfblasExecute
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasExecute (
        unsigned int kernelIndex = 0,
        unsigned int deviceIndex = 0
        )

This function starts the kernel and wait until it finishes.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80
    
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for instrution


2.3.16 xfblasExecuteAsync
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    void xfblasExecuteAsync (
        unsigned int numKernels = 1,
        unsigned int deviceIndex = 0
        )

This asynchronous function starts all kernels and wait until them finish.


.. rubric:: Parameters:

.. list-table::
    :widths: 20 80
    
    *
        - numKernels
        - number of kernels that is being used, default is 1
    *
        - deviceIndex
        - index of device that is being used, default is 0

2.3.17 xfblasGetByPointer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasGetByPointer (
        void* A,
        unsigned int kernelIndex = 0,
        unsigned int deviceIndex = 0
        )

This function copies a matrix in FPGA device memory to host memory by pointer.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - A
        - pointer to matrix A in the host memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix

2.3.18 xfblasGetByAddress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasGetByAddress (
        void* A,
        unsigned long long p_bufSize,
        unsigned int offset,
        unsigned int kernelIndex = 0,
        unsigned int deviceIndex = 0
        )

This function copies a matrix in FPGA device memory to host memory by its address in device memory.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - A
        - pointer to matrix A in the host memory
    *
        - p_bufSize
        - size of matrix A
    *
        - offset
        - A's address in device memory
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if there is no FPGA device memory allocated for the matrix

        
2.4 Vitis BLAS Function Reference
-----------------------------------

2.4.1 xfblasGemm
^^^^^^^^^^^^^^^^^^

.. code-block:: cpp
    :class: title-code-block

    xfblasStatus_t xfblasGemm(xfblasOperation_t transa, xfblasOperation_t transb, int m, int n, int k, int alpha, void* A, int lda, void* B, int ldb, int beta, void* C, int ldc, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0)

This function performs the matrix-matrix multiplication C = alpha*op(A)op(B) + beta*C. See L3 examples for detail usage.

.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - transa
        - operation op(A) that is non- or (conj.) transpose
    *
        - transb
        - operation op(B) that is non- or (conj.) transpose
    *
        - m
        - number of rows in matrix A, matrix C
    *
        - n
        - number of cols in matrix B, matrix C
    *
        - k
        - number of cols in matrix A, number of rows in matrix B
    *
        - alpha
        - scalar used for multiplication
    *
        - A
        - pointer to matrix A in the host memory
    *
        - lda
        - leading dimension of matrix A
    *
        - B
        - pointer to matrix B in the host memory
    *
        - ldb
        - leading dimension of matrix B
    *
        - beta
        - scalar used for multiplication
    *
        - C
        - pointer to matrix C in the host memory
    *
        - ldc
        - leading dimension of matrix C
    *
        - kernelIndex
        - index of kernel that is being used, default is 0
    *
        - deviceIndex
        - index of device that is being used, default is 0
        
.. rubric:: Return:

.. list-table::
    :widths: 20 80
    
    *
        - xfblasStatus_t
        - 0 if the operation completed successfully
    *
        - xfblasStatus_t
        - 1 if the library was not initialized
    *
        - xfblasStatus_t
        - 3 if not all the matrices have FPGA devie memory allocated
    *
        - xfblasStatus_t
        - 4 if the engine is not supported for now
 
3. Obtain FPGA bitstream 
=========================
FPGA bitstreams could be built in examples or tests folder by using command make build TARGET=hw PLATFORM_REPO_PATHS=LOCAL_PLATFORM_PATH

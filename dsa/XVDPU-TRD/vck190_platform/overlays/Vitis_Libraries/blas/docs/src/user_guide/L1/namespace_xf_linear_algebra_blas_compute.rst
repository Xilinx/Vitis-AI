
.. meta::
   :keywords: BLAS, Library, Vitis BLAS Library, namespace, bias, compute
   :description: Vitis BLAS library namespace xf::blas.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. index:: pair: namespace; xf::blas
.. _doxid-namespacexf_1_1blas:

namespace xf::blas
==================

.. toctree::
	:hidden:

Overview
~~~~~~~~










.. index:: pair: function; gbmv
.. _doxid-namespacexf_1_1blas_1abe4199bfd663b774cba0b9eb27a03e08:
.. index:: pair: function; gemv
.. _doxid-namespacexf_1_1blas_1a8757bb7d347fa70520f7dfb0bf5015ea:
.. index:: pair: function; gemv
.. _doxid-namespacexf_1_1blas_1a4b8faeb6b0cf62275d81118ee9950585:
.. index:: pair: function; symv
.. _doxid-namespacexf_1_1blas_1ab6d6cccceb77ca9d8b8a28fc510d2021:
.. index:: pair: function; trmv
.. _doxid-namespacexf_1_1blas_1a97b563c3509866438a45cbde63270435:




.. ref-code-block:: cpp
	:class: overview-code-block

	
	namespace blas {

	// global functions

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType>
	void :ref:`amax<doxid-namespacexf_1_1blas_1ac97d95595ee23916874cd96abaebb5e6>`(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_IndexType& p_result);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType>
	void :ref:`amin<doxid-namespacexf_1_1blas_1ab41dc9640742e1bd31336510e383c8b6>`(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_IndexType& p_result);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void :ref:`asum<doxid-namespacexf_1_1blas_1a9352151f4d43f9f20ae9cc1e34630159>`(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_DataType& p_sum);

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void :ref:`axpy<doxid-namespacexf_1_1blas_1a61a3e6dcf46aaf5ddbf0d1b79dfc5a10>`(unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y, hls::stream<WideType<t_DataType, t_ParEntries>>& p_r);

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void :ref:`copy<doxid-namespacexf_1_1blas_1ab97952c9808d350040356d44d5902eb1>`(unsigned int p_n, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void :ref:`dot<doxid-namespacexf_1_1blas_1a8af71f19450d24aa338a30b225cdcac0>`(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_y, t_DataType& p_res);

	template  <typename t_DataType, unsigned int t_ParEntries, unsigned int t_MaxRows, typename t_IndexType = unsigned int, typename t_MacType = t_DataType>
	void gbmv(const unsigned int p_m, const unsigned int p_n, const unsigned int p_kl, const unsigned int p_ku, hls::stream<WideType<t_DataType, t_ParEntries>>& p_A, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_MacType, t_ParEntries>>& p_y);

	template  <typename t_DataType, unsigned int t_ParEntries, unsigned int t_MaxRows, typename t_IndexType = unsigned int, typename t_MacType = t_DataType>
	void :ref:`gbmv<doxid-namespacexf_1_1blas_1a5cec2775b5d537226ba858df112e6587>`(const unsigned int p_m, const unsigned int p_n, const unsigned int p_kl, const unsigned int p_ku, const t_DataType p_alpha, hls::stream<WideType<t_DataType, t_ParEntries>>& p_M, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y, hls::stream<WideType<t_DataType, t_ParEntries>>& p_yr);

	template  <typename t_DataType, unsigned int t_LogParEntries, unsigned int t_NumStreams = (1 << t_LogParEntries), typename t_IndexType = unsigned int>
	void gemv(const unsigned int p_m, const unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>> p_M [t_NumStreams], hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>> p_x [t_NumStreams], hls::stream<WideType<t_DataType, t_NumStreams>>& p_y);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void gemv(const unsigned int p_m, const unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_M, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, hls::stream<WideType<t_DataType, 1>>& p_y);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void :ref:`gemv<doxid-namespacexf_1_1blas_1aa4b14aa75c1be1d8a90d0cafa7f3e279>`(const unsigned int p_m, const unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_M, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType, 1>>& p_y, hls::stream<WideType<t_DataType, 1>>& p_yr);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void :ref:`nrm2<doxid-namespacexf_1_1blas_1a0b72b37d89ee2c2b5b0e2881aaa35595>`(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_DataType& p_res);

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void :ref:`scal<doxid-namespacexf_1_1blas_1a2f58b1501c5999577f4e4b64d4c4f285>`(unsigned int p_n, t_DataType p_alpha, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_res);

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void :ref:`swap<doxid-namespacexf_1_1blas_1af375ccc98b8aa70a9f0af3dbeb44e8f5>`(unsigned int p_n, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y, hls::stream<WideType<t_DataType, t_ParEntries>>& p_xRes, hls::stream<WideType<t_DataType, t_ParEntries>>& p_yRes);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void symv(const unsigned int p_n);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void :ref:`symv<doxid-namespacexf_1_1blas_1a3185b8d5eb180275476cef08612b82dc>`(const unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_M, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_y, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_yr);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int, typename t_MacType = t_DataType>
	void trmv(const bool uplo, const unsigned int p_n);

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void :ref:`trmv<doxid-namespacexf_1_1blas_1a49d777b37946c19c6f84604af3650a3f>`(const bool uplo, const unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_M, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType, 1>>& p_y, hls::stream<WideType<t_DataType, 1>>& p_yr);

	} // namespace blas
.. _details-doxid-namespacexf_1_1blas:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

amax
#####

.. index:: pair: function; amax
.. _doxid-namespacexf_1_1blas_1ac97d95595ee23916874cd96abaebb5e6:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType>
	void amax(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_IndexType& p_result)

amax function that returns the position of the vector element that has the maximum magnitude.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in the input vector p_x, p_n % l_ParEntries == 0

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_result

		- the resulting index, which is 0 if p_n <= 0

amin
#####

.. index:: pair: function; amin
.. _doxid-namespacexf_1_1blas_1ab41dc9640742e1bd31336510e383c8b6:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType>
	void amin(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_IndexType& p_result)

amin function that returns the position of the vector element that has the minimum magnitude.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in the input vector p_x, p_n % l_ParEntries == 0

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_result

		- the resulting index, which is 0 if p_n <= 0

asum
#####

.. index:: pair: function; asum
.. _doxid-namespacexf_1_1blas_1a9352151f4d43f9f20ae9cc1e34630159:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void asum(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_DataType& p_sum)

asum function that returns the sum of the magnitude of vector elements.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in the input vector p_x, p_n % l_ParEntries == 0

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_sum

		- the sum, which is 0 if p_n <= 0

axpy
#####

.. index:: pair: function; axpy
.. _doxid-namespacexf_1_1blas_1a61a3e6dcf46aaf5ddbf0d1b79dfc5a10:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void axpy(unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y, hls::stream<WideType<t_DataType, t_ParEntries>>& p_r)

axpy function that compute Y = alpha*X + Y.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in the input vector p_x, p_n % t_ParEntries == 0

	*
		- p_x

		- the input stream of packed entries of vector X

	*
		- p_y

		- the input stream of packed entries of vector Y

	*
		- p_r

		- the output stream of packed entries of result vector Y

copy
#####

.. index:: pair: function; copy
.. _doxid-namespacexf_1_1blas_1ab97952c9808d350040356d44d5902eb1:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void copy(unsigned int p_n, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y)

copy function that compute Y = X



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_ParEntries

		- number of parallelly processed entries in the packed input vector stream

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in vector X and Y

	*
		- p_x

		- the packed input vector stream

	*
		- p_y

		- the packed output vector stream

dot
####

.. index:: pair: function; dot
.. _doxid-namespacexf_1_1blas_1a8af71f19450d24aa338a30b225cdcac0:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void dot(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_y, t_DataType& p_res)

dot function that returns the dot product of vector x and y.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in the input vector p_x, p_n % l_ParEntries == 0

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_res

		- the dot product of x and y

gbmv
#####

.. index:: pair: function; gbmv
.. _doxid-namespacexf_1_1blas_1a5cec2775b5d537226ba858df112e6587:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_ParEntries, unsigned int t_MaxRows, typename t_IndexType = unsigned int, typename t_MacType = t_DataType>
	void gbmv(const unsigned int p_m, const unsigned int p_n, const unsigned int p_kl, const unsigned int p_ku, const t_DataType p_alpha, hls::stream<WideType<t_DataType, t_ParEntries>>& p_M, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y, hls::stream<WideType<t_DataType, t_ParEntries>>& p_yr)

gbmv function performs general banded matrix-vector multiplication matrix and a vector y = alpha * M * x + beta * y



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_ParEntries

		- the number of parallelly processed entries in the input vector

	*
		- t_MaxRows

		- the maximum size of buffers for output vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- t_MacType

		- the datatype of the output stream

	*
		- p_m

		- the number of rows of input matrix p_M

	*
		- p_alpha

		- scalar alpha

	*
		- p_M

		- the input stream of packed Matrix entries

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_beta

		- scalar beta

	*
		- p_y

		- the output vector

gemv
#####


.. index:: pair: function; gemv
.. _doxid-namespacexf_1_1blas_1aa4b14aa75c1be1d8a90d0cafa7f3e279:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void gemv(const unsigned int p_m, const unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_M, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType, 1>>& p_y, hls::stream<WideType<t_DataType, 1>>& p_yr)

gemv function that returns the result vector of the multiplication of a matrix and a vector y = alpha * M * x + beta * y



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_m

		- the number of rows of input matrix p_M

	*
		- p_n

		- the number of cols of input matrix p_M, as well as the number of entries in the input vector p_x, p_n % l_ParEntries == 0

	*
		- p_alpha

		- scalar alpha

	*
		- p_M

		- the input stream of packed Matrix entries

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_beta

		- scalar beta

	*
		- p_y

		- the output vector

nrm2
#####

.. index:: pair: function; nrm2
.. _doxid-namespacexf_1_1blas_1a0b72b37d89ee2c2b5b0e2881aaa35595:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void nrm2(unsigned int p_n, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, t_DataType& p_res)

nrm2 function that returns the Euclidean norm of the vector x.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in the input vector p_x, p_n % (1<<l_LogParEntries) == 0

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_res

		- the nrm2 of x

scal
#####

.. index:: pair: function; scal
.. _doxid-namespacexf_1_1blas_1a2f58b1501c5999577f4e4b64d4c4f285:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void scal(unsigned int p_n, t_DataType p_alpha, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_res)

scal function that compute X = alpha * X



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_ParEntries

		- number of parallelly processed entries in the packed input vector stream

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in vector X, p_n % t_ParEntries == 0

	*
		- p_x

		- the packed input vector stream

	*
		- p_res

		- the packed output vector stream

swap
#####

.. index:: pair: function; swap
.. _doxid-namespacexf_1_1blas_1af375ccc98b8aa70a9f0af3dbeb44e8f5:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
	void swap(unsigned int p_n, hls::stream<WideType<t_DataType, t_ParEntries>>& p_x, hls::stream<WideType<t_DataType, t_ParEntries>>& p_y, hls::stream<WideType<t_DataType, t_ParEntries>>& p_xRes, hls::stream<WideType<t_DataType, t_ParEntries>>& p_yRes)

swap function that swap vector x and y



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_ParEntries

		- number of parallelly processed entries in the packed input vector stream

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of entries in vector X and Y, p_n % t_ParEntries == 0

	*
		- p_x

		- the packed input vector stream

	*
		- p_y

		- the packed input vector stream

	*
		- p_xRes

		- the packed output stream

	*
		- p_yRes

		- the packed output stream

symv
#####

.. index:: pair: function; symv
.. _doxid-namespacexf_1_1blas_1a3185b8d5eb180275476cef08612b82dc:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void symv(const unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_M, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_y, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_yr)

symv function that returns the result vector of the multiplication of a symmetric matrix and a vector y = alpha * M * x + beta * y



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the dimention of input matrix p_M, as well as the number of entries in the input vector p_x, p_n % l_ParEntries == 0

	*
		- p_alpha

		- 

	*
		- scalar

		- alpha

	*
		- p_M

		- the input stream of packed Matrix entries

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_beta

		- 

	*
		- scalar

		- beta

	*
		- p_y

		- the output vector

trmv
#####

.. index:: pair: function; trmv
.. _doxid-namespacexf_1_1blas_1a49d777b37946c19c6f84604af3650a3f:

.. ref-code-block:: cpp
	:class: title-code-block

	template  <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
	void trmv(const bool uplo, const unsigned int p_n, const t_DataType p_alpha, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_M, hls::stream<WideType<t_DataType,(1<<t_LogParEntries)>>& p_x, const t_DataType p_beta, hls::stream<WideType<t_DataType, 1>>& p_y, hls::stream<WideType<t_DataType, 1>>& p_yr)

trmv function that returns the result vector of the multiplication of a triangular matrix and a vector y = alpha * M * x + beta * y



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t_DataType

		- the data type of the vector entries

	*
		- t_LogParEntries

		- log2 of the number of parallelly processed entries in the input vector

	*
		- t_IndexType

		- the datatype of the index

	*
		- p_n

		- the number of cols of input matrix p_M, as well as the number of entries in the input vector p_x, p_n % l_ParEntries == 0

	*
		- p_alpha

		- 

	*
		- scalar

		- alpha

	*
		- p_M

		- the input stream of packed Matrix entries

	*
		- p_x

		- the input stream of packed vector entries

	*
		- p_beta

		- 

	*
		- scalar

		- beta

	*
		- p_y

		- the output vector


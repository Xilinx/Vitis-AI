.. index:: pair: namespace; graph
.. _doxid-namespacexf_1_1graph:
.. _cid-xf::graph:

namespace graph
===============

.. toctree::
	:hidden:

	namespace_xf_graph_enums.rst



.. _doxid-namespacexf_1_1graph_1a3aedeb50dfe1fb8fcc38dc1a4565baa2:
.. _cid-xf::graph::pagerankcore:
.. _doxid-namespacexf_1_1graph_1a54304bcae353c4edd7b8d51440ce4c10:
.. _cid-xf::graph::preprocessdata:
.. ref-code-block:: cpp
	:class: overview-code-block

	// namespaces

	namespace :ref:`xf::graph::enums<doxid-namespacexf_1_1graph_1_1enums>`
	namespace :ref:`xf::graph::internal<doxid-namespacexf_1_1graph_1_1internal>`
	namespace :ref:`xf::graph::internal::dense_similarity<doxid-namespacexf_1_1graph_1_1internal_1_1dense__similarity>`
	namespace :ref:`xf::graph::internal::general_similarity<doxid-namespacexf_1_1graph_1_1internal_1_1general__similarity>`
	namespace :ref:`xf::graph::internal::sort_top_k<doxid-namespacexf_1_1graph_1_1internal_1_1sort__top__k>`
	namespace :ref:`xf::graph::internal::sparse_similarity<doxid-namespacexf_1_1graph_1_1internal_1_1sparse__similarity>`


.. FunctionSection

.. _doxid-namespacexf_1_1graph_1aee23e30ff149bd6af6cabddc358e8326:
.. _cid-xf::graph::densesimilarity:

denseSimilarity
---------------


.. code-block:: cpp
	
	#include "similarity/dense_similarity.hpp"



.. ref-code-block:: cpp
	:class: title-code-block

	template <
	    int CHNM,
	    int PU,
	    int WData,
	    int RAM_SZ,
	    bool EN_FLOAT_POINT
	    >
	void denseSimilarity (
	    hls::stream <ap_uint <32>>& config,
	    hls::stream <ap_uint <WData>>& sourceWeight,
	    hls::stream <ap_uint <WData*CHNM>> strmIn0 [PU],
	    hls::stream <ap_uint <WData*CHNM>> strmIn1 [PU],
	    hls::stream <ap_uint <WData*CHNM>> strmIn2 [PU],
	    hls::stream <ap_uint <WData*CHNM>> strmIn3 [PU],
	    hls::stream <ap_uint <WData>>& rowID,
	    hls::stream <float>& similarity,
	    hls::stream <bool>& strmOutEnd
	    )

similarity function for dense graph. It support both Jaccard and Cosine Similarity.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - CHNM

        - the channel number of input data

    *
        - PU

        - the number of processing unit

    *
        - WData

        - the width of input data

    *
        - RAM_SZ

        - the log size of internal URAM

    *
        - EN_FLOAT_POINT

        - if it is true, the primitive will support both float and int type of input. Otherwise, it only support int. Multiple channel of float input should be compacted as type of ap_uint.

    *
        - config

        - the control parameter of the primitive which contains: sourceNUM, similarityType, dataType, startID, rowNUM and colNUM of each processing unit(PU)

    *
        - sourceWeight

        - input weight as source for computing similarity

    *
        - strmIn0

        - input muti-channel data stream for PU0

    *
        - strmIn1

        - input muti-channel data stream for PU1

    *
        - strmIn2

        - input muti-channel data stream for PU2

    *
        - strmIn3

        - input muti-channel data stream for PU3

    *
        - rowID

        - output result ID stream

    *
        - similarity

        - output similarity value corresponding to its ID

    *
        - strmOutEnd

        - end flag stream for output

.. _doxid-namespacexf_1_1graph_1aee23e30ff149bd6af6cabddc358e8327:
.. _cid-xf::graph::densesimilarity:

denseSimilarity
---------------


.. code-block:: cpp
	
	#include "similarity/dense_similarity_int.hpp"



.. ref-code-block:: cpp
	:class: title-code-block

	template <
	    int CHNM,
	    int PU,
	    int WData,
	    int RAM_SZ,
	    >
	void denseSimilarity (
	    hls::stream <ap_int <32>>& config,
	    hls::stream <ap_int <WData>>& sourceWeight,
	    hls::stream <ap_int <WData*CHNM>> strmIn0 [PU],
	    hls::stream <ap_int <WData*CHNM>> strmIn1 [PU],
	    hls::stream <ap_int <WData*CHNM>> strmIn2 [PU],
	    hls::stream <ap_int <WData*CHNM>> strmIn3 [PU],
	    hls::stream <ap_int <WData>>& rowID,
	    hls::stream <float>& similarity,
	    hls::stream <bool>& strmOutEnd
	    )

similarity function for dense graph. It support both Jaccard and Cosine Similarity.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - CHNM

        - the channel number of input data

    *
        - PU

        - the number of processing unit

    *
        - WData

        - the width of input data

    *
        - RAM_SZ

        - the log size of internal URAM

    *
        - config

        - the control parameter of the primitive which contains: sourceNUM, similarityType, dataType, startID, rowNUM and colNUM of each processing unit(PU)

    *
        - sourceWeight

        - input weight as source for computing similarity

    *
        - strmIn0

        - input muti-channel data stream for PU0

    *
        - strmIn1

        - input muti-channel data stream for PU1

    *
        - strmIn2

        - input muti-channel data stream for PU2

    *
        - strmIn3

        - input muti-channel data stream for PU3

    *
        - rowID

        - output result ID stream

    *
        - similarity

        - output similarity value corresponding to its ID

    *
        - strmOutEnd

        - end flag stream for output



.. _doxid-namespacexf_1_1graph_1a74b52499352ebd940934c5a145449ea3:
.. _cid-xf::graph::generalsimilarity:

generalSimilarity
-----------------


.. code-block:: cpp
	
	#include "similarity/general_similarity.hpp"



.. ref-code-block:: cpp
	:class: title-code-block

	template <
	    int CHNM,
	    int PU,
	    int WData,
	    int RAM_SZ,
	    bool EN_FLOAT_POINT
	    >
	void generalSimilarity (
	    hls::stream <ap_uint <32>>& config,
	    hls::stream <ap_uint <WData>>& sourceIndice,
	    hls::stream <ap_uint <WData>>& sourceWeight,
	    hls::stream <ap_uint <WData*CHNM>> strmIn0 [PU],
	    hls::stream <ap_uint <WData*CHNM>> strmIn1 [PU],
	    hls::stream <ap_uint <WData*CHNM>> strmIn2 [PU],
	    hls::stream <ap_uint <WData*CHNM>> strmIn3 [PU],
	    hls::stream <ap_uint <WData>>& rowID,
	    hls::stream <float>& similarity,
	    hls::stream <bool>& strmOutEnd
	    )

similarity function which support both dense and sparse graph. It also support both Jaccard and Cosine Similarity.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - CHNM

        - the channel number of input data

    *
        - PU

        - the number of processing unit

    *
        - WData

        - the width of input data

    *
        - RAM_SZ

        - the log size of internal URAM

    *
        - EN_FLOAT_POINT

        - if it is true, the primitive will support both float and int type of input. Otherwise, it only support int. Multiple channel of float input should be compacted as type of ap_uint.

    *
        - config

        - the control parameter of the primitive which contains: sourceNUM, similarityType, graphType, dataType, startID, rowNUM and colNUM of each processing unit(PU)

    *
        - sourceIndice

        - input indice as source for computing similarity

    *
        - sourceWeight

        - input weight as source for computing similarity

    *
        - strmIn0

        - input muti-channel data stream for PU0

    *
        - strmIn1

        - input muti-channel data stream for PU1

    *
        - strmIn2

        - input muti-channel data stream for PU2

    *
        - strmIn3

        - input muti-channel data stream for PU3

    *
        - rowID

        - output result ID stream

    *
        - similarity

        - output similarity value corresponding to its ID

    *
        - strmOutEnd

        - end flag stream for output

.. _doxid-namespacexf_1_1graph_1a6cb25e98a52faae5a63c6c4a24d8250b:
.. _cid-xf::graph::sorttopk:

sortTopK
--------


.. code-block:: cpp
	
	#include "similarity/sort_top_k.hpp"



.. ref-code-block:: cpp
	:class: title-code-block

	template <
	    typename KEY_TYPE,
	    typename DATA_TYPE,
	    int MAX_SORT_NUMBER
	    >
	void sortTopK (
	    hls::stream <DATA_TYPE>& dinStrm,
	    hls::stream <KEY_TYPE>& kinStrm,
	    hls::stream <bool>& endInStrm,
	    hls::stream <DATA_TYPE>& doutStrm,
	    hls::stream <KEY_TYPE>& koutStrm,
	    hls::stream <bool>& endOutStrm,
	    int k,
	    bool order
	    )

sort top k function.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - KEY_TYPE

        - the input and output key type

    *
        - DATA_TYPE

        - the input and output data type

    *
        - MAX_SORT_NUMBER

        - the max number of the sequence can be sorted

    *
        - dinStrm

        - input data stream

    *
        - kinStrm

        - input key stream

    *
        - endInStrm

        - end flag stream for input

    *
        - doutStrm

        - output data stream

    *
        - koutStrm

        - output key stream

    *
        - endOutStrm

        - end flag stream for output

    *
        - number

        - of top K

    *
        - order

        - 1:sort ascending 0:sort descending

.. _doxid-namespacexf_1_1graph_1a43bc588c252f4307bc5dbf37def60c4e:
.. _cid-xf::graph::sparsesimilarity:

sparseSimilarity
----------------


.. code-block:: cpp
	
	#include "similarity/sparse_similarity.hpp"



.. ref-code-block:: cpp
	:class: title-code-block

	template <
	    int CHNM,
	    int PU,
	    int WData,
	    int RAM_SZ,
	    bool EN_FLOAT_POINT
	    >
	void sparseSimilarity (
	    hls::stream <ap_uint <32>>& config,
	    hls::stream <ap_uint <WData>>& sourceIndice,
	    hls::stream <ap_uint <WData>>& sourceWeight,
	    hls::stream <ap_uint <WData*CHNM>> offsetCSR [PU],
	    hls::stream <ap_uint <WData*CHNM>> indiceCSR [PU],
	    hls::stream <ap_uint <WData*CHNM>> weight [PU],
	    hls::stream <ap_uint <WData>>& rowID,
	    hls::stream <float>& similarity,
	    hls::stream <bool>& strmOutEnd
	    )

similarity function for sparse graph. It support both Jaccard and Cosine Similarity.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - CHNM

        - the channel number of input data

    *
        - PU

        - the number of processing unit

    *
        - WData

        - the width of input data

    *
        - RAM_SZ

        - the log size of internal URAM

    *
        - EN_FLOAT_POINT

        - if it is true, the primitive will support both float and int type of input. Otherwise, it only support int. Multiple channel of float input should be compacted as type of ap_uint.

    *
        - config

        - the control parameter of the primitive which contains: sourceNUM, similarityType, dataType, startID, rowNUM and colNUM of each processing unit(PU)

    *
        - sourceIndice

        - input indice as source vertex for computing similarity

    *
        - sourceWeight

        - input weight as source vertex for computing similarity

    *
        - offsetCSR

        - input muti-channel offset stream

    *
        - indiceCSR

        - input muti-channel indice stream

    *
        - weight

        - input muti-channel weight stream

    *
        - rowID

        - output result ID stream

    *
        - similarity

        - output similarity value corresponding to its ID

    *
        - strmOutEnd

        - end flag stream for output


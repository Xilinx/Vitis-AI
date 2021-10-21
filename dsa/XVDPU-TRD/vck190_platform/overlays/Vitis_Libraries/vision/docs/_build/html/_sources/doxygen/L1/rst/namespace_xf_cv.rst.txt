.. index:: pair: namespace; cv
.. _doxid-namespacexf_1_1cv:
.. _cid-xf::cv:

namespace cv
============

.. toctree::
	:hidden:



.. ref-code-block:: cpp
	:class: overview-code-block


.. FunctionSection




.. _doxid-namespacexf_1_1cv_1acd442e9091610a362b520eb14c563928:
.. _cid-xf::cv::insertborder:

insertBorder
------------


.. code-block:: cpp
	
	#include "dnn/xf_insertBorder.hpp"



.. ref-code-block:: cpp
	:class: title-code-block

	template <
	    int TYPE,
	    int SRC_ROWS,
	    int SRC_COLS,
	    int DST_ROWS,
	    int DST_COLS,
	    int NPC
	    >
	void insertBorder (
	    xf::cv::Mat <TYPE, SRC_ROWS, SRC_COLS, NPC>& _src,
	    xf::cv::Mat <TYPE, DST_ROWS, DST_COLS, NPC>& _dst,
	    int insert_pad_val
	    )



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - TYPE

        - input and ouput type

    *
        - SRC_ROWS

        - rows of the input image

    *
        - SRC_COLS

        - cols of the input image

    *
        - DST_ROWS

        - rows of the output image

    *
        - DST_COLS

        - cols of the output image

    *
        - NPC

        - number of pixels processed per cycle

    *
        - _src

        - input image

    *
        - _dst

        - output image

    *
        - insert_pad_val

        - insert pad value


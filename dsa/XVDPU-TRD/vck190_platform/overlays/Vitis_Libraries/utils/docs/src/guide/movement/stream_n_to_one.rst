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

.. _guide-stream_n_to_one:

*****************************************
Internals of streamNToOne
*****************************************

The :ref:`streamNToOne <cid-xf::common::utils_hw::streamNToOne>` API
is designed for collecting data from multiple processor units.
Three different algorithms have been implemented, ``RoundRobinT``,
``LoadBalanceT`` and ``TagSelectT``.

To ensure the throughput, it is very common to pass a vector of elements in
FPGA data paths, so ``streamNToOne`` supports element vector output, if the
data elements are passed in the form of ``ap_uint``.
It also offers overload for generic template type for non-vector output.

.. contents::
   :depth: 2

Round-Robin
===========

The round-robin algorithm collects elements from input streams in circular
order, starting from the output stream with index 0.

Generic Type
~~~~~~~~~~~~

With generic type input, the function dispatches one element per cycle.
This mode works best for sharing the multi-cycle processing work across
an array of units.

.. image:: /images/stream_n_to_one_round_robin_type.png
   :alt: structure of round-robin collect
   :width: 80%
   :align: center

Vector Output
~~~~~~~~~~~~~

The design of the primitive includes 3 modules:

1. fetch: attempt to read data from the `n` input streams.

2. vectorize: Inner buffers as wide as the least common multiple of ``N * Win``
   and ``Wout`` are used to combine the inputs into vectors.

3. emit: read vectorized data and emit to output stream.

.. image:: /images/stream_n_to_one_round_robin_detail.png
   :alt: structure of vectorized round-robin collection
   :width: 100%
   :align: center

.. ATTENTION::
   Current implementation has the following limitations:

   * It uses a wide ``ap_uint`` as internal buffer. The buffer is as wide as
     the least common multiple (LCM) of input width and total output width.
     The width is limited by ``AP_INT_MAX_W``, which defaults to 1024.
   * This library will try to override ``AP_INT_MAX_W`` to 4096, but user
     should ensure that ``ap_int.h`` has not be included before the library
     headers.
   * Too large ``AP_INT_MAX_W`` will significantly slow down HLS synthesis.


Load-Balancing
==============

The load-balancing algorithm does not keep a fixed order in collection,
instead, it skips predecessors that cannot be read, and tries to feed as much
as possible to output.

Generic Type
~~~~~~~~~~~~

.. image:: /images/stream_n_to_one_load_balance_type.png
   :alt: structure of load-balance collection
   :width: 80%
   :align: center


Vector Output
~~~~~~~~~~~~~~

The design of the primitive includes 3 modules:

1. fetch: attempt to read data from the `n` input streams.

2. vectorize: Inner buffers as wide as the least common multiple of  ``N * Win``
   and ``Wout`` are used to combine the inputs into vectors.

3. emit: read vectorized data and emit to output stream.

.. image:: /images/stream_n_to_one_load_balance_detail.png
   :alt: structure of vectorized load-balance collection
   :width: 100%
   :align: center

.. ATTENTION::
   Current implementation has the following limitations:

   * It uses a wide ``ap_uint`` as internal buffer. The buffer is as wide as
     the least common multiple (LCM) of input width and total output width.
     The width is limited by ``AP_INT_MAX_W``, which defaults to 1024.
   * This library will try to override ``AP_INT_MAX_W`` to 4096, but user
     should ensure that ``ap_int.h`` has not be included before the library
     headers.
   * Too large ``AP_INT_MAX_W`` will significantly slow down HLS synthesis.

.. IMPORTANT::
   The depth of output streams must be no less than 4 due to internal delay.

Tag-Select
==========

This algorithm collects data elements according to provided tags.
The tags are used as index of input streams, and it is expected that
each input element is accompanied by a tag.

.. image:: /images/stream_n_to_one_tag_select_type.png
   :alt: structure of tag-select collect
   :width: 80%
   :align: center


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

.. _stream_based:

***************************************************
Stream-Based API Design
***************************************************

.. toctree::
   :maxdepth: 1

Stream-based Interface
======================

The interfaces of the primitives in this library are mostly HLS streams, with a 1-bit flag stream
along with the main data stream throughout the dataflow region.

.. code:: cpp

        hls::stream<ap_uint<W> >& data_strm,
        hls::stream<bool>&        e_data_strm,

The packet data protocol in stream-based design is illustrated in the following figure.
For each valid data present in ``data_strm``, a ``false`` value would present the corresponding
``e_data_strm``. Meanwhile, an appended ``true`` value has to be given to close this packet.
So, stream consumer can be notified when data transfer is over. And for a given packet,
the number of elements in ``e_data_strm`` would be always one more than in corresponding
``data_strm`` during each transaction.

.. image:: /images/stream_based_protocol.png
   :alt: The protocl of stream packet data
   :scale: 80%
   :align: center

The benefits of this interface are

* Within a HLS dataflow region, all primitives connected via HLS streams can work in
  parallel, and this is the key to FPGA acceleration.

* Using the 1-bit flag stream to mark *end of operation* can trigger stream consumer
  as soon as the first row data becomes available, without known how many rows will be
  generated later. Moreover, it can represent empty table.



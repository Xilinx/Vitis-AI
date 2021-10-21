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
   :keywords: Merge-Join, Merge-Left-Join, mergeJoin, mergeLeftJoin, combineCol, mergeJoin, mergeLeftJoin
   :description: Describes the structure and execution of Merge-Join and Merge-Left-Join.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-merge_join:

************************************************
Internals of Merge-Join and Merge-Left-Join 
************************************************

.. toctree::
   :hidden:
   :maxdepth: 1

This document gives the user guide and describes the structure of Merge-Join and Merge-Left-Join, implemented as :ref:`mergeJoin<cid-xf::database::mergeJoin>` function and :ref:`mergeLeftJoin<cid-xf::database::mergeLeftJoin>` function respectively.

----------
User guide
----------
        When calling the ``mergeJoin``/``mergeLeftJoin`` function, users need to set the key type and payload type. Only one key stream and one payload stream is given for an input table. If multiple key columns or multiple payload columns are required, please use the :ref:`combineCol <cid-xf::database::combineCol>` to combine columns.

        The user needs to push input tables into the related streams. Users also need to configure the function to merge ascend/descend tables by setting the ``isascend`` parameter to true/false.

.. CAUTION:: The left table should not contain duplicated keys.
..

        The left and right result tables are pushed out in separate stream. If required, please use the :ref:`combineCol <cid-xf::database::combineCol>` to combine left and right table into one stream.

.. Important:: The mergeLeftJoin function has a isnull_strm output stream to mark if the result right table is null (The current left key does not exist in the right table).

---------
Structure
---------
        Use the merge join of ascend tables as an example:

.. image:: /images/merge_join.png
   :alt: Merge Join Structure
   :align: center
..

Every clock cycle, compare the keys from left and right tables:

        - If the keys are not the same, pull the stream with a smaller key and no output.
        - If the keys are the same, pull the right stream and push the keys and payloads to the output stream.

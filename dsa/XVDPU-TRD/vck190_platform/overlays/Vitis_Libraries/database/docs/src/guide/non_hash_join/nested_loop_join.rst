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
   :keywords: Nested-Loop-Join, nestedLoopJoin
   :description: Describes the structure and execution of Nested-Loop-Join.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-nested_loop_join:

**********************************
Internals of Nested-Loop-Join
**********************************

.. toctree::
   :hidden:
   :maxdepth: 1

This document gives the user guide and describes the structure of the Nested-Loop-Join, implemented as :ref:`nestedLoopJoin<cid-xf::database::nestedLoopJoin>` function.

User guide
----------

        When calling the ``nestedLoopJoin`` function, users need to set the key type and payload type. Only one key stream and one payload stream is given for an input table. If multiple key columns or multiple payload columns are required, please use the :ref:`combineCol <cid-xf::database::combineCol>` to combine columns.

        Every left row will become an independent channel to compare with the right table. Users need to set the number of channels by setting the ``CMP_NUM`` template parameter. 50 is a typical number for the ``CMP_NUM``. 

.. CAUTION:: Very large CMP_NUM (more than 120) may result in numerous resource.
..

        Users need to push the left and right tables into the associated streams. The number of rows of the left table should not exceed the predefined ``CMP_NUM``. But it can be less than the ``CMP_NUM``. Unused channels will generate an empty table (assert the end of table flag for one cycle) to the next module.

Structure
---------

.. image:: /images/nested_loop_join.png
   :alt: Nested Loop Join Structure
   :align: center
..

The following steps will be performed when the nested loop join function is called:

        - Load the left table using shift registers. 
        - Pull out right table row by row and compared with the left table.

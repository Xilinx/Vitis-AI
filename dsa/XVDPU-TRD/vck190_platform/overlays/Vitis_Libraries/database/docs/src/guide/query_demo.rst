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
   :keywords: Vitis Database Library, acceleration, demo
   :description: Vitis Database Library application acceleration demos.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _demo_app:

Query-Specific Acceleration Demo
================================

The following application acceleration demos are shipped within this library in ``demo_app_accel`` folder.

.. contents:: List of Demos
   :local:
   :depth: 1

TPC-H Query 5 Simplified
------------------------

This demo in ``demo_app_accel/q5_simplified_100g`` directory
implements the following query using TPC-H test data.

.. code-block:: sql

   select
           sum(l_extendedprice * (1 - l_discount)) as revenue
   from
           orders,
           lineitem
   where
           l_orderkey = o_orderkey
           and o_orderdate >= date '1994-01-01'
           and o_orderdate < date '1995-01-01'
   ;



TPC-H Query 5
-------------

This demo in ``demo_app_accel/q5`` directory implements TPC-H Query 5 with scale factor 1 data. 
The SQL being implemented is

.. code-block:: sql

   select
           n_name,
           sum(l_extendedprice * (1 - l_discount)) as revenue
   from
           customer,
           orders,
           lineitem,
           supplier,
           nation,
           region
   where
           c_custkey = o_custkey
           and l_orderkey = o_orderkey
           and l_suppkey = s_suppkey
           and c_nationkey = s_nationkey
           and s_nationkey = n_nationkey
           and n_regionkey = r_regionkey
           and r_name = 'MIDDLE EAST'
           and o_orderdate >= date '1994-01-01'
           and o_orderdate < date '1995-01-01'
   group by
           n_name
   order by
           revenue desc
   ;


TPC-H Query 6 Modified
----------------------

This demo in ``demo_app_accel/q6_mod`` directory implements TPC-H Query 6 modified with
extra conditions. The SQL being implemented is

.. code-block:: sql

   select
   	sum(l_extendedprice * l_discount) as revenue
   from
   	lineitem
   where
   	l_shipdate >= date '1994-01-01' and l_shipdate < date '1995-01-01'
   	and l_discount between 0.06 - 0.01 and 0.06 + 0.01
   	and l_quantity < 24
   	and l_shipdate > l_commitdate;
   ;

.

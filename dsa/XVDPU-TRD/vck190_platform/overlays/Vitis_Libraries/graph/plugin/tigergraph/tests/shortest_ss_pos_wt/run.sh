#!/bin/bash
gadmin stop -fy
gadmin start -fy
gsql schema_xgraph.gsql
gsql load_xgraph.gsql
gsql ../../algorithms/shortest_ss_pos_wt_fpga.gsql
gsql query.gsql
gsql -g xgraph "SET query_timeout=240000000 run query test_shortest_ss_pos_wt_fpga()"


#!/bin/bash
gadmin stop -fy
gadmin start -fy
gsql schema_xgraph.gsql
gsql load_xgraph.gsql
gsql ../../algorithms/bfs_fpga.gsql
gsql query.gsql
gsql -g xgraph "SET query_timeout=240000000 run query test_bfs_fpga()"


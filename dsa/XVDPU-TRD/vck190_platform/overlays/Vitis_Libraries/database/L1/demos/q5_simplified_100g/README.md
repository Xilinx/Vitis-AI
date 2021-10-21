# Vitis Project for TPC-H Q5 Simplified

## Overview

This project demostrate FPGA acceleration of the following query, simplified from Query 5, over TPC-H data of scale factor 1, 3, 10, 30 and 100.

```
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
```

## Running the Benchmark

Usage can be queried with `make help`. Basic use is:

```
make run TARGET=sw_emu DEVICE=/path/to/xpfm
```

Change `sw_emu` to `hw_emu` or `hw` to run RTL simulation or board test correspondingly.


## Dataset

We used the TPC-H dataset generated with ssb-dbgen tool.
The host binary will download the tool and create data files in the same folder as itself.


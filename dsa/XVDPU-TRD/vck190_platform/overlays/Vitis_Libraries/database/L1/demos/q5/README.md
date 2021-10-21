# Vitis Project for TPC-H Q5

## Overview

This project demostrate FPGA acceleration of TPC-H Q5 on scale factor 1 data.

```
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


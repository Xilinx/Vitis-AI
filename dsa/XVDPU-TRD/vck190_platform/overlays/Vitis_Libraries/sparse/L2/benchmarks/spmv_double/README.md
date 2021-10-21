spmv_double
=========

This example resides in ``L2/benchmarks/spmv_double`` directory. The tutorial provides a step-by-step guide that covers commands for building and running the benchmark.

Executable Usage
-----------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/sparse/L2/benchmarks#building). For getting the design,

```
   cd L2/benchmarks/spmv_double
```

* **Build hardware and host executable(Step 2)** 

Run the following make commands to build your XCLBIN and host binary targeting Alveo U280. Please note that the hardware building process will take a long time. It can be 8-10 hours.

```
    make build TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_201920_3
    make host TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_201920_3
```

* **Run benchmark(Step 3)**

To get the benchmark results, please run the following command.

**Generate test inputs:**

```
    conda activate xf_blas
    source ./gen_test.sh
```
The gen_test.sh triggers a set of python scripts to download the .mtx files listed in test.txt under current directory and partitions them evenly across 16 HBM channels. Each partitioned data set, including the value and indices of each NNZ entry, is stored in one HBM channel. Each row of the partitioned data set is padded to multiple of 32 to accommodate the double precision accumulation latency. The padding overhead for each matrix is summarized in the benchmark result as well. This overhead will be reduced with the improvement of floating point support on FPGA platforms.

**Run tests:**

```
    python ./run_test.py
```
The run_test.py launches the host executable with each partitioned data set and offloads the double precision SpMV operation to U280 card. The SpMV operation is run numerous time (2000 in this benchmark) to mask out the host code overhead. The total run time in the benchmark results includes the OpenCl function call time to trigger the CUs and the hardware run time. The run time [ms] / iteration field gives single SpMV run time on the U280 card.

* **Example output(Step 4)** 

```
    All tests pass!
    Please find the benchmark results in spmv_perf.csv.
```

Profiling
----------

The xclbin can be built at 250 MHz. The data type supported in the hardware is double precision float64 type.
The hardware resource utilization and benchmark results are shown in the two tables below.

##### Table 1 Hardware resources

| Name                | LUT               | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
|---------------------|-------------------|------------------|-------------------|----------------|---------------|----------------|
| Platform            |  165475 [ 12.70%] |  31045 [  5.17%] |  285828 [ 10.97%] |  323 [ 16.02%] |  64 [  6.67%] |    4 [  0.04%] |
| User Budget         | 1137245 [100.00%] | 569435 [100.00%] | 2319612 [100.00%] | 1693 [100.00%] | 896 [100.00%] | 9020 [100.00%] |
|    Used Resources   |  220980 [ 19.43%] |  30170 [  5.30%] |  336240 [ 14.50%] |  211 [ 12.46%] |  64 [  7.14%] |  900 [  9.98%] |
|    Unused Resources |  916265 [ 80.57%] | 539265 [ 94.70%] | 1983372 [ 85.50%] | 1482 [ 87.54%] | 832 [ 92.86%] | 8120 [ 90.02%] |

##### Table 2 Benchmark results

|matrix        | rows| cols| NNZs  |padded rows| padded cols| padded NNZs| padding ratio| num of runs| total run time[sec]| time[ms]/run|
|--------------|-----|-----|-------|-----------|------------|------------|--------------|------------|--------------------|-------------|
|nasa2910      |2910 |2910 |174296 |2910       |2912        |297952      |1.70946       |2000        |0.102513            |0.0512565    |
|ex9           |3363 |3363 |99471  |3363       |3364        |199328      |2.00388       |2000        |0.0759525           |0.0379762    |
|bcsstk24      |3562 |3562 |159910 |3562       |3564        |222656      |1.39238       |2000        |0.0747713           |0.0373857    |
|bcsstk15      |3948 |3948 |117816 |3948       |3948        |267488      |2.27039       |2000        |0.0872443           |0.0436221    |
|bcsstk28      |4410 |4410 |219024 |4410       |4412        |319264      |1.45767       |2000        |0.116322            |0.0581609    |
|s3rmt3m3      |5357 |5357 |207695 |5357       |5360        |330624      |1.59187       |2000        |0.106942            |0.0534711    |
|s2rmq4m1      |5489 |5489 |281111 |5489       |5492        |427648      |1.52128       |2000        |0.126217            |0.0631087    |
|nd3k          |9000 |9000 |3279690|9000       |9000        |4277792     |1.30433       |2000        |0.677946            |0.338973     |
|ted_B_unscaled|10605|10605|144579 |10605      |10608       |548416      |3.79319       |2000        |0.136411            |0.0682054    |
|ted_B         |10605|10605|144579 |10605      |10608       |548416      |3.79319       |2000        |0.149135            |0.0745673    |
|msc10848      |10848|10848|1229778|10848      |10848       |2050720     |1.66755       |2000        |0.391394            |0.195697     |
|cbuckle       |13681|13681|676515 |13681      |13684       |924832      |1.36705       |2000        |0.216792            |0.108396     |
|olafu         |16146|16146|1015156|16146      |16148       |1452320     |1.43064       |2000        |0.263899            |0.131949     |
|gyro_k        |17361|17361|1021159|17361      |17364       |1932384     |1.89234       |2000        |0.412774            |0.206387     |
|bodyy4        |17546|17546|121938 |17546      |17548       |710112      |5.82355       |2000        |0.269815            |0.134907     |
|nd6k          |18000|18000|6897316|18000      |18000       |9415552     |1.3651        |2000        |1.50509             |0.752544     |
|raefsky4      |19779|19779|1328611|19779      |19780       |2268704     |1.70758       |2000        |0.446744            |0.223372     |
|bcsstk36      |23052|23052|1143140|23052      |23052       |1833056     |1.60353       |2000        |0.374293            |0.187146     |
|msc23052      |23052|23052|1154814|23052      |23052       |3093824     |2.67907       |2000        |0.723612            |0.361806     |
|ct20stif      |52329|52329|2698463|52329      |52332       |4846624     |1.79607       |2000        |1.01894             |0.509468     |
|nasasrb       |54870|54870|2677324|54870      |54872       |4856256     |1.81385       |2000        |0.780656            |0.390328     |
|bodyy6        |19366|19366|134748 |19366      |19368       |762688      |5.66011       |2000        |0.247517            |0.123759     |

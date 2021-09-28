#!/usr/bin/env bash

# Copyright 2019 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ "$#" -ne 3 ]; then
	echo "Usage: $0 <thread#> <data_type> <mode>" >&2
	echo "where <data_type> = float double" >&2
	echo "<mode> = (g)enerate binary, (b)enchmark, (a)ll" >&2
	exit 1
fi

DATA_TYPE=$2
MODE=$3

NUMA="numactl -i all"
export OMP_NUM_THREADS=$1


if [[ ("$MODE" != "g") && ("$MODE" != "b") && ("$MODE" != "a") ]]; then
	echo "Error in mode"
	exit 1
fi

if [[ ("$MODE" == "g") || ("$MODE" == "a") ]]; then
	# Build
	if [[ ("$DATA_TYPE" == "double") ]]; then
		make dgemm_mkl_gen
	elif [[ ("$DATA_TYPE" == "float") ]]; then
		 make sgemm_mkl_gen
	elif [[ ("$DATA_TYPE" == "short") ]]; then
		make sgemm_mkl_gen_short

	else
		echo "Error in data_type"
		exit 1
	fi
	
	if [ ! -e "../data/$DATA_TYPE" ]; then
		mkdir -p ../data/$DATA_TYPE
	fi
	# Run
	n=256
	while [  $n -le 8192 ]; do
		echo "############# $n ################"
		if [[ ("$DATA_TYPE" == "double") ]]; then
			if [ -e dgemm_mkl_gen ]; then
				./dgemm_mkl_gen $n $n $n ../data/$DATA_TYPE/
			else
				echo "Error in Generating Binary: ./dgemm_mkl_gen not found"
				exit 1
			fi
		elif [[ ("$DATA_TYPE" == "float") ]]; then
			if [ -e sgemm_mkl_gen ]; then
				./sgemm_mkl_gen $n $n $n ../data/$DATA_TYPE/
			else
				echo "Error in Generating Binary: ./sgemm_mkl_gen not found"
				exit 1
			fi
		elif [[ ("$DATA_TYPE" == "short") ]]; then
			if [ -e sgemm_mkl_gen_short ]; then
				./sgemm_mkl_gen_short $n $n $n ../data/$DATA_TYPE/
			else
				echo "Error in Generating Binary: ./short_gemm_mkl_gen not found"
				exit 1
			fi
		else
			echo "Error in data_type"
			exit 1
		fi
		n=`expr $n \* 2`
	done
	echo "====================="
	echo "Generating binary file (Golden data) complete"
	echo "Binary file is at ../data/$DATA_TYPE/"
	echo "====================="
fi

if [[ ("$MODE" == "b") || ("$MODE" == "a") ]]; then
	# Build
	if [[ ("$DATA_TYPE" == "double") ]]; then
		make dgemm_mkl_bench
	elif [[ ("$DATA_TYPE" == "float") ]]; then
		 make sgemm_mkl_bench
	elif [[ ("$DATA_TYPE" == "short") ]]; then
		echo "Benchmarking Error: datatype (short) is not supported in MKL"
		exit 1
	else
		echo "Benchmarking Error in data_type"
		exit 1
	fi

	# Run
	n=256
	logs=()
	while [  $n -le 8192 ]; do
		echo "############# $n ################"
		if [[ ("$DATA_TYPE" == "double") ]]; then
			if [ -e dgemm_mkl_bench ]; then
				./dgemm_mkl_bench $n $n $n | tee log-$DATA_TYPE-$n.txt
			else
				echo "Error in Benchmarking: ./dgemm_mkl_bench not found"
				exit 1
			fi
		elif [[ ("$DATA_TYPE" == "float") ]]; then
			if [ -e sgemm_mkl_bench ]; then
				./sgemm_mkl_bench $n $n $n | tee log-$DATA_TYPE-$n.txt
			else
				echo "Error in Benchmarking: ./sgemm_mkl_bench not found"
				exit 1
			fi
		else
			echo "Benchmarking Error in data_type"
			exit 1
		fi
		logs="$logs log-$DATA_TYPE-$n.txt"
		n=`expr $n \* 2`
	done
	echo "====================="
	echo "GEMM MKL Benchmarking complete"
        cat /proc/cpuinfo | grep "model name" | head -1 | tr ':' ',' > perf_gemm_mkl_bench.csv
	egrep -h ^DATA_CSV $logs | grep Type | head -1 >> perf_gemm_mkl_bench.csv
	egrep -h ^DATA_CSV $logs | grep -v Type >> perf_gemm_mkl_bench.csv
	echo "Parsing CSV complete"
	echo "====================="
fi

echo "Done!"

exit 0

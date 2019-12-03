#!/usr/bin/env bash
# Copyright 2019 Xilinx Inc.
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

usage() {
    echo "Usage: ./test.sh [options]"
    echo "pytest args"
    echo "    -k <STR>: run specific tests matching this substring"
    echo "    -s: disable capturing stdout in pytest. Set this option to enable printing"
    echo "    --cov: enable coverage measurement. Requires coverage package"
    echo "custom args"
    echo "    --hostname <STR>: hostname of Neptune server. Defaults to localhost"
    echo "    --port <NUM>: port of Neptune server. Defaults to 8998"
    echo "    --wsport <NUM>: port of Neptune websocket server. Defaults to 8999"
    echo "    --fpgas <NUM>: number of FPGAs available for testing. Defaults to 0"
    echo "    --benchmark <skip|only|all>: skip or only run benchmark tests, or run everything. Defaults to skip"
    echo "    --neptune_mode <n|d|q|c|qc>: don't start neptune or start Neptune with default flags (no flags) or the quiet/ clean flags or both. Defaults to qc"
}

HOSTNAME="localhost"
PORT=8998
WSPORT=8999
TESTS=""
CAPTURE=""
FPGAS=0
BENCHMARK="skip"
MODE="qc"
COV=""

while true
do
    case "$1" in
        --hostname     ) HOSTNAME="$2"    ; shift 2 ;;
        --port         ) PORT="$2"        ; shift 2 ;;
        --wsport       ) WSPORT="$2"      ; shift 2 ;;
        --fpgas        ) FPGAS=$2         ; shift 2 ;;
        --benchmark    ) BENCHMARK=$2     ; shift 2 ;;
        --neptune_mode ) MODE="$2"        ; shift 2 ;;
        --cov          ) COV="--coverage" ; shift 1 ;;
        -k             ) TESTS="$2"       ; shift 2 ;;
        -s             ) CAPTURE="-s"     ; shift 1 ;;
        -h |--help     ) usage            ; exit  1 ;;
        *) break ;;
    esac
done

if [[ $TESTS != "" ]]; then
    TESTS="-k $TESTS"
fi

. ../overlaybins/setup.sh > /dev/null # needed to add VAI_ALVEO_ROOT to env
export PYTHONPATH=${PYTHONPATH}$VAI_ALVEO_ROOT:
pytest $CAPTURE -rA --hostname $HOSTNAME --port $PORT --wsport $WSPORT \
    --fpgas $FPGAS --benchmark $BENCHMARK --neptune_mode $MODE $TESTS $COV
if [[ $COV == "--coverage" ]]; then
    coverage combine
    coverage report -m

    # in the absence of CI, update coverage badge if coverage-badge exists
    if [ -x "$(command -v coverage-badge)" ]; then
        coverage-badge -o ./docs/coverage.svg -f
    fi
fi

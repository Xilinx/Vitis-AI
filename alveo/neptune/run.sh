#!/bin/bash

usage() {
    echo "Usage:"
    echo "./run.sh"
    echo "    --port: port to run Neptune server on. Defaults to 8998"
    echo "    --wsport: port to run websocket on. Defaults to 8999"
    echo "    -q/--quiet: suppresses printing"
    echo "    --clean: cleans directory of temporary files"
    echo "    --cov: enables coverage measurement"
    echo "    --help: prints this message"
}

QUIET=0
CLEAN=0
COV=0
PORT=8998
WSPORT=8999

while true
do
    case "$1" in
        --port      ) PORT="$2"  ; shift 2 ;;
        --wsport     ) WSPORT="$2" ; shift 2 ;;
        -q |--quiet ) QUIET=1    ; shift 1 ;;
        --clean     ) CLEAN=1    ; shift 1 ;;
        --cov       ) COV=1      ; shift 1 ;;
        -h |--help  ) usage      ; exit  1 ;;
        *) break ;;
    esac
done

if [[ $CLEAN == 1 ]]; then
    find . -name '*.pyc' -delete
    rm -f ./recipes/recipe_*.bak
    rm -f /tmp/xsnodes/ipc_handles/*
fi

if [[ $COV == 1 ]]; then
    cmd="coverage run"
else
    cmd="python"
fi

if [[ $QUIET == 1 ]]; then
    . ../overlaybins/setup.sh &> /dev/null
    export PYTHONPATH=${PYTHONPATH}$VAI_ALVEO_ROOT:
    $cmd server.py --port=$PORT --wsport=$WSPORT &> /dev/null
else
    . ../overlaybins/setup.sh
    export PYTHONPATH=${PYTHONPATH}$VAI_ALVEO_ROOT:
    $cmd server.py --port=$PORT --wsport=$WSPORT
fi

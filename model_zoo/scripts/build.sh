#!/bin/bash

build_if_not_exists() {
    EVAL_SCRIPT_FOLDER=$1
    EVAL_APP=$2
    cd $EVAL_SCRIPT_FOLDER
    if [ -e "$EVAL_APP" ]; then
        echo "Run $EVAL_APP"
    else
        echo "Run build.sh"
        bash build.sh
        echo "Run $EVAL_APP"
    fi
}

#!/bin/bash

source scripts/config.env

build_if_not_exists() {
    EVAL_APP=$1
    cd $EVAL_SCRIPT_FOLDER
    if [ -e "$EVAL_APP" ]; then
        echo "Run $EVAL_APP"
    else
        echo "Run build.sh"
        bash build.sh
        echo "Run $EVAL_APP"
    fi
}

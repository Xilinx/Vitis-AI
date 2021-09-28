#!/usr/bin/env bash
L1_tests=L1/tests
L2_tests=L2/tests
L2_demos=L2/demos
L3_demos=L3/demos

declare -a run_designs=($L1_tests $L2_demos $L2_tests $L3_demos)

for designs in "${run_designs[@]}"
do
    cd $designs; 
    echo "Executing $designs designs"
    for d in */ ; do
        cd $d;
        filename=$1
        if [[ -f "$filename" ]]; then
            if [ $designs == $L1_tests ]; then
                make run CSIM=1 XPART=xcu200
                make cleanall
            else
                make run TARGET=sw_emu  
                make cleanall
            fi
            echo "$d"
        fi
        cd ..
    done
    cd ../../
done

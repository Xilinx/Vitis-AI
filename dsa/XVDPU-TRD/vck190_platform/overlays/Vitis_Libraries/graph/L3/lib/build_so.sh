
xrtPath=/opt/xilinx/xrt
xrmPath=/opt/xilinx/xrm

while getopts ":r:m:" opt
do
case $opt in
    r)
    xrtPath=$OPTARG
    echo "$xrtPath"
    ;;
    m)
    xrmPath=$OPTARG
    echo "$xrmPath"
    ;;
    ?)
    echo "unknown"
    exit 1;;
    esac
done

source $xrtPath/setup.sh

source $xrmPath/setup.sh

make clean

make libgraphL3


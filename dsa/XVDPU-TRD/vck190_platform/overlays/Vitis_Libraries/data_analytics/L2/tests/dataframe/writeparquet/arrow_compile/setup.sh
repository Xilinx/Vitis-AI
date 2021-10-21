set -ex
PWD=`pwd`
ext=$PWD/arrow_compile
COMPILER_PATH=/usr/bin
if [ ! -z "$1" ];then
COMPILER_PATH=$1
fi

# install cmake
# In ubuntu18.04 defalut cmake(3.10)
# In ubuntu16.04 defalut cmake(3.2) 
if [ ! -f ${ext}/cmake-3.18.0-rc1-Linux-x86_64.sh ];then
wget https://cmake.org/files/v3.18/cmake-3.18.0-rc1-Linux-x86_64.sh -P ${ext}/
chmod 755 ${ext}/cmake-3.18.0-rc1-Linux-x86_64.sh
mkdir -p ${ext}/cmake-3.18.0
${ext}/cmake-3.18.0-rc1-Linux-x86_64.sh --skip-license --exclude-subdir --prefix=${ext}/cmake-3.18.0
fi
export PATH=`pwd`/arrow_compile/cmake-3.18.0/bin:$PATH

if [ -d ${ext}/arrow ];then
echo "arrow has exist, use the directory"
else
git clone https://github.com/apache/arrow.git ${ext}/arrow
cd ${ext}/arrow
git checkout tags/apache-arrow-3.0.0
fi

# install arrow
echo "building arrow..."
if [ ! -d ${ext}/arrow/cpp/release ];then
mkdir -p ${ext}/arrow/cpp/release
fi
cd ${ext}/arrow/cpp/release
cmake .. -DCMAKE_CXX_COMPILER=$COMPILER_PATH/g++ -DCMAKE_C_COMPILER=$COMPILER_PATH/gcc -DCMAKE_BUILD_TYPE=release -DARROW_JSON:BOOL=ON -DARROW_CSV:BOOL=ON  -DARROW_USE_OPENSSL=ON -DARROW_PARQUET=ON -DPARQUET_REQUIRE_ENCRYPTION=ON -DARROW_DEPENDENCY_SOURCE=BUNDLED 
#cmake .. -DARROW_FLIGHT=ON -DARROW_BUILD_TESTS=ON -DARROW_DEPENDENCY_SOURCE=BUNDLED
make -j 8
exit 0



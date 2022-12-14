g++ -o memperftool ./noc/noc.cpp ./xapm/apm.cpp  mem_perf_test.cpp -lpthread -I./include -I./noc/include -I./xapm/include
g++ -shared ./noc/noc.cpp ./xapm/apm.cpp mem_perf_interface.cpp -fPIC -o libmemperf.so -I./include -I./noc/include -I./xapm/include

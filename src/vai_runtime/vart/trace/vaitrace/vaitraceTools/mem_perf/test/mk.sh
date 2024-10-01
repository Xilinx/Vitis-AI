g++ -o memperftool ../noc/ddrmc/noc.cpp ../noc/nmu/noc_nmu.cpp ../xapm/apm.cpp  mem_perf_test.cpp -lpthread -I../mem_base/include -I../noc/ddrmc/include -I../noc/nmu/include -I../xapm/include
g++ -shared    ../noc/ddrmc/noc_ddrmc.cpp ../xapm/apm.cpp ../mem_interface/mem_perf_interface.cpp -fPIC -o libmemperf.so -I../mem_mon_base/include  -I../noc/ddrmc/include -I../xapm/include
g++ -shared  ../noc/nmu/noc_nmu.cpp   ../mem_interface/nmu_perf_interface.cpp -fPIC -o libnmuperf.so -I../mem_mon_base/include  -I../noc/nmu/include

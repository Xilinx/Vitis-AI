g++ -o powertool  power_mon.cpp power_mon_test.cpp -I./include
g++ -shared power_mon.cpp power_mon_test.cpp -fPIC -o libpowermon.so -I./include

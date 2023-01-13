g++  -o apmtool apm.cpp apm_test.cpp  -lpthread -I./include -I../include
g++ -shared apm.cpp apm_shell.cpp -fPIC -o libxapm.so -I./include -I../include

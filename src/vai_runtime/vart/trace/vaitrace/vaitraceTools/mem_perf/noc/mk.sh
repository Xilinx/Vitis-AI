g++ -o noctool  noc.cpp  noc_test.cpp  -lpthread -I./include -I../includ
g++ -shared noc.cpp noc_shell.cpp -fPIC -o libnoc.so -I./include -I../include
 

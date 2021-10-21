## Level 3: Python Host software APIs

This directory contains python examples for the various financial models.
These use python modules which are created using PyBind11 to connect to the software APIS.
The file module.cpp contains all these modules, written in C++.  To generate the modules run the Makefile - make
To use copy the desired example python script and xclbin file into the generated output subdirectory
and run from that directory - for example python36 ./dje_test.py
Note within each example the card type is defined and there is a comment describing the expected result.

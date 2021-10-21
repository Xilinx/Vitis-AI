OVERVIEW:
The AIE DSP library example design. 
Project includes an example design of a 129 tap single rate symmetric FIR. It's purpose is to ilustrate a DSPLIB element instantiation in user's graphs, including graph initialization with argument passing.

USAGE: 
Compile the example design with AIECompiler, using: 
    make compile

To simulate, invoke AIEsimulator with: 
    make sim
This will generate the file output.txt in aiesimulator_output folder.

To compare the output results with golden file reference, extract samples from output.txt, then do a diff with respect to the reference, using the below command:
    make check_op

Generate status summary with: 
    make get_status

Display status.txt to console with: 
    make summary

All the above steps are performed in sequence when default target is called: 
    make all

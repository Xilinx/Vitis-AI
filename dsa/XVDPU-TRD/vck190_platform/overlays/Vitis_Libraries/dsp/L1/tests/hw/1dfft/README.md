# Input/output sample order (1D-FFT)

Unlike the traditional FFT where the input/output vectors are given in 1-dimensional, we introduce the super sample rate (SSR) in our HLS FFT design for boosting the FPGA acceleration, so the input/output vectors are transformed into arrays for FPGA can easily consume the input samples within a single column in 1 cycle.

Let's take a 16 points (the shortest length allowed for HLS FFT) FFT for example:

If we describe the 1-D input vector as:


a | b | c | d | e | f | g | h | i | j | k | l | m | n | o | p
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---

The result:

A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---

If the SSR is set to 2, the corresponding input array (`inData[SSR][FFT_LEN / SSR]`) should be like:

col0 | col1 | col2 | col3 | col4 | col5 | col6 | col7
--- | --- | --- | --- | --- | --- | --- | ---
a | c | e | g | i | k | m | o
b | d | f | h | j | l | n | p

The output array from HLS FFT shoud be:

col0 | col1 | col2 | col3 | col4 | col5 | col6 | col7
--- | --- | --- | --- | --- | --- | --- | ---
A | C | E | G | I | K | M | O
B | D | F | H | J | L | N | P

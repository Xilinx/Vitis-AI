# Input/output sample order (2D-FFT)

As the 2D-FFT works by taking a 1-dimentional FFT along the columns of the 2-dimentional input matrix, and produces a 2-dimentional matrix of 1-dimentional Fourier coefficients, then perfoms a 1-dimentional FFT on the rows of the generated coefficients matrix. Also, the super sample rate (SSR) is handled within the HLS 2-D FFT module, the input array for HLS 2D FFT is 2-dimentional array with the same order as the traditional 2-D FFT.

Let's take a 16x16 points (the shortest length allowed for HLS FFT) 2-D FFT for example:

If we describe the 2-D input matrix (`inMat[row][col]`) as:

col0 | col1 | col2 | col3 
--- | --- | --- | --- 
a | b | c | d
e | f | g | h
i | j | k | l
m | n | o | p


The result 2-D matrix should be:

col0 | col1 | col2 | col3 
--- | --- | --- | --- 
A | B | C | D
E | F | G | H
I | J | K | L
M | N | O | P

# Unit Tests for different Floating Point Single Precision FFT Configurations
Unit tests for different primitive floating point ( Single Precision) FFT Configuration.These tests are HLS project. They verify the functionality of different unique FFT configuration which leads to different architecture and provide the II, latency and resource utilization after csynthesis/HLS Synthesis.
## Impulse Test
Runs an impulse test on FFT, to launch it cahnge to folder "impulse_test" , set_env and launch  "run_test.tcl" using vitis_hls
1. cd impulse_test
2. source set_env.sh
3. vitis_hls -f run_test.tcl
## FFT 1D Test Suite "fft_1d_snr"
This folder contains a number test for FFT 1-D case. It will run test and also calculate signal to noise ratio at the output. To launch test follow the steps as follows:
1. cd fft_1d_snr
2. source run_all_test.sh

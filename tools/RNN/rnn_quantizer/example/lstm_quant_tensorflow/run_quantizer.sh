#!/bin/bash

python quantize_lstm.py --quant_mode calib --subset_len 1000
python quantize_lstm.py --quant_mode test --subset_len 1000

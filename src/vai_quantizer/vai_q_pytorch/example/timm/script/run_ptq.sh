# Calibration
python validate_quant.py  --quantized_out quantize_result/ptq --quant_mode calib --data-dir $2 -b 32 --model $1 --subset_len 5120
# Test
python validate_quant.py  --quantized_out quantize_result/ptq --quant_mode test  --data-dir $2 -b 32 --model $1
# Deployment
python validate_quant.py  --quantized_out quantize_result/ptq --quant_mode test  --data-dir $2 -b 1 --model $1 --subset_len 1 --deploy
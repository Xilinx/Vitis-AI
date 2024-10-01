# Calibration with Fast_finetune
python validate_quant.py --quantized_out quantize_result/fast_finetune --quant_mode calib --data-dir $2 -b 32 --model $1 --fast_finetune --ff_subset_len 1024 --subset_len 5120
# Test with Fast_finetune
python validate_quant.py --quantized_out quantize_result/fast_finetune --quant_mode test  --data-dir $2 -b 32 --model $1 --fast_finetune
# Deployment with Fast_finetune
python validate_quant.py --quantized_out quantize_result/fast_finetune --quant_mode test  --data-dir $2 -b 1  --model $1 --fast_finetune --subset_len 1 --deploy

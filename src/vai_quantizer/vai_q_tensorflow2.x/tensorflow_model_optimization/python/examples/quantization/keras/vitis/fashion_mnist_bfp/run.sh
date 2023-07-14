export CUDA_VISIBLE_DEVICES='0'

python main.py --batch_size 120 \
  --epochs 5 \
  --qat_epochs 2 \
  --bit_width 13 \
  --save_dir ./result \
  --origin_model_file origin.h5 \
  --eager_mode False \
  --quantize True \
  --quantize_strategy bfp \
  --data_format msfp \
  --use_qat True


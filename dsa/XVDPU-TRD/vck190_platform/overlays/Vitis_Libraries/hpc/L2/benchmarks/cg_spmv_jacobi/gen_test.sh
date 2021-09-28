rm -rf ./sig_dat
rm -rf ./vec_dat
python ../../../../sparse/L2/tests/fp64/spmv/python/gen_signature.py  --partition --mtx_list ./test.txt --sig_path ./sig_dat
python ../../../../sparse/L2/tests/fp64/spmv/python/gen_signature.py  --check --mtx_list ./test.txt --sig_path ./sig_dat
python ../../../../sparse/L2/tests/fp64/spmv/python/gen_vectors.py  --gen_vec --pcg --mtx_list ./test.txt --vec_path ./vec_dat

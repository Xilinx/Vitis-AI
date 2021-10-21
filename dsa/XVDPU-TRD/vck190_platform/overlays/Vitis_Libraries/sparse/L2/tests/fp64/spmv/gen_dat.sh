rm -rf ./par_res
rm -rf ./sig_dat
rm -rf ./vec_dat
python ./python/gen_partition.py --partition --mtx_list ./test.txt --par_path ./par_res
python ./python/gen_partition.py --check --mtx_list ./test.txt --par_path ./par_res
python ./python/gen_signature.py --gen_sig --verbose --par_path ./par_res --sig_path ./sig_dat
python ./python/gen_signature.py --check --par_path ./par_res --sig_path ./sig_dat
python ./python/gen_vectors.py --verbose --gen_vec --mtx_list ./test.txt --vec_path ./vec_dat

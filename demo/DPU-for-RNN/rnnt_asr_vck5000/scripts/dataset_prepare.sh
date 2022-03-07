mkdir my_work_dir/local_data
mv dev-clean.tar.gz my_work_dir/local_data/
cd my_work_dir/local_data/
tar -xzvf dev-clean.tar.gz
cd ../../ ;sh scripts/dataset_preprocess.sh


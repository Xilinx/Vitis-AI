#!/bin/bash


wget https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.1.zip
mv openDownload?filename=all_models_1.1.zip all_models_1.1.zip
check_result=`md5sum -c <<<"10bedfa99692c5d0e7f840d23d0cd8d0 all_models_1.1.zip"`
#echo $check_result

if [ "$check_result" != "all_models_1.1.zip: OK" ]; then
   echo "md5sum check failed! Please try to download again."
   exit 1
else
   if [ `command -v unzip` ]; then
      unzip all_models_1.1.zip -d models
   else 
      sudo apt install unzip
      unzip all_models_1.1.zip -d models
   fi  
   echo "all models downloaded successfully."
   exit 0
fi




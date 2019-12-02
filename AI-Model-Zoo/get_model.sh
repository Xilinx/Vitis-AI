#!/bin/bash


wget https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.0.zip
mv openDownload?filename=all_models_1.0.zip all_models_1.0.zip
check_result=`md5sum -c <<<"ed5509bcd0ce5e3aa2b220145acc17f5 all_models_1.0.zip"`
#echo $check_result

if [ "$check_result" != "all_models_1.0.zip: OK" ]; then
   echo "md5sum check failed! Please try to download again."
   exit 1
else
   if [ `command -v unzip` ]; then
      unzip all_models_1.0.zip -d models
   else 
      sudo apt install unzip
      unzip all_models_1.0.zip -d models
   fi  
   echo "all models downloaded successfully."
   exit 0
fi




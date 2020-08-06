#!/bin/bash


wget https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.2.zip
mv openDownload?filename=all_models_1.2.zip all_models_1.2.zip
check_result=`md5sum -c <<<"6766149c79e762a97d8d9c004a216ffe all_models_1.2.zip"`
#echo $check_result

if [ "$check_result" != "all_models_1.2.zip: OK" ]; then
   echo "md5sum check failed! Please try to download again."
   exit 1
else
   if [ `command -v unzip` ]; then
      unzip all_models_1.2.zip -d models
   else 
      sudo apt install unzip
      unzip all_models_1.2.zip -d models
   fi  
   echo "all models downloaded successfully."
   exit 0
fi




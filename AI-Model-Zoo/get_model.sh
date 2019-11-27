#!/bin/bash


wget https://www.xilinx.com/bin/public/openDownload?filename=all_models.zip
mv openDownload?filename=all_models.zip all_models.zip
check_result=`md5sum -c <<<"0fc242102699cad110027ecfff453d91 all_models.zip"`
#echo $check_result

if [ "$check_result" != "all_models.zip: OK" ]; then
   echo "md5sum check failed! Please try to download again."
   exit 1
else
   if [ `command -v unzip` ]; then
      unzip all_models.zip -d models
   else 
      sudo apt install unzip
      unzip all_models.zip -d models
   fi  
   echo "all models downloaded successfully."
   exit 0
fi




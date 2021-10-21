if [ -z $1 ]; then
    echo "Number of devices argument missing"
    exit 1
fi

if [[ -n ${1//[0-9]/} ]]; then
    echo "Device count must be an integer"
    exit 1
fi
if [ $1 -eq 0 ]; then
    echo "No of devices must be greater than zero"
    exit 1
fi

rm -rf /tmp/xrm_$USER.json
DEVICES=$1
printf "{\n" >> /tmp/xrm_$USER.json
printf "\t\"request\": {\n" >> /tmp/xrm_$USER.json
printf "\t\t\"name\":\"load\",\n" >> /tmp/xrm_$USER.json
printf "\t\t\"requestId\":1,\n" >> /tmp/xrm_$USER.json
printf "\t\t\"parameters\":[\n" >> /tmp/xrm_$USER.json

for ((i = 0 ; i < $1 ; i++))
do 
printf "\t\t\t{\n" >> /tmp/xrm_$USER.json
printf "\t\t\t\t\"device\":%s,\n" "$i" >> /tmp/xrm_$USER.json
printf "\t\t\t\t\"xclbin\":\"%s\"\n" "$XILINX_LIBZ_XCLBIN" >> /tmp/xrm_$USER.json
printf "\t\t\t}\n" >> /tmp/xrm_$USER.json
if [ ${DEVICES} > 1 ] 
then
    VAR=$(expr $i + 1)
    if [ ${DEVICES} != $VAR ] 
    then
        printf "\t\t\t,\n" >> /tmp/xrm_$USER.json
    fi
fi
done


printf "\t\t]\n" >> /tmp/xrm_$USER.json
printf "\t}\n" >> /tmp/xrm_$USER.json
printf "}\n" >> /tmp/xrm_$USER.json
chmod 777 /tmp/xrm_$USER.json
xrmadm /tmp/xrm_$USER.json

sudo cp /workspace/xclbin/u50lv/dpu.xclbin /usr/lib/dpu.xclbin

mkdir -p /workspace/app/imdb_sentiment_detection/model
mkdir -p /workspace/app/customer_satisfaction/model
mkdir -p /workspace/app/open_information_extraction/model
cp /workspace/models/u50lv/imdb_sentiment_detection/*  /workspace/app/imdb_sentiment_detection/model 
cp /workspace/models/u50lv/customer_satisfaction/*  /workspace/app/customer_satisfaction/model 
cp /workspace/models/u50lv/open_information_extraction/*  /workspace/app/open_information_extraction/model 
mv /workspace/app/open_information_extraction/model/weights.th  /workspace/app/open_information_extraction/weights

echo "set HW and models successfully"

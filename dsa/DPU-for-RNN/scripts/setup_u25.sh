sudo cp /workspace/xclbin/u25/dpu.xclbin /usr/lib/dpu.xclbin

cp /workspace/models/u25/imdb_sentiment_detection/*  /workspace/app/imdb_sentiment_detection/model 
cp /workspace/models/u25/customer_satisfaction/*  /workspace/app/customer_satisfaction/model 
cp /workspace/models/u25/open_information_extraction/*  /workspace/app/open_information_extraction/model 
mv /workspace/app/open_information_extraction/model/weights.th  /workspace/app/open_information_extraction/weights

echo "set HW and models successfully"

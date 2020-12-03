## Vitis AI RNN Tool

Vitis AI RNN tool is designed to support different RNN networks, including RNN, LSTM, Bi-directional LSTM and GRU. 
Currently only standard LSTM is supported.

Vitis AI RNN tool includes rnn quantizer and rnn compiler. 
The quantizer performs fixed-point int16 quantization for model parameters and input. 
The compiler generates instructions based on the target network structure and RNN hardware architecture. 
So far, this tool chain only supports standard LSTM, and the generated instructions can only be deployed in U25 card and U50 card. 
For U25 card, the weights of model can’t exceed 4MB, and for U50 card, the weights can’t exceed 8MB. 
The workflow of the tool chain shows as follows:

1. Import RNN model and input argument to RNN quantizer, get the quantization information of the model.

2. Using the quantization information, RNN quantizer converts float model to quantized model. Get the Xmodel file, which is the representation of the model’s computation process.

3. RNN compiler compiles the Xmodel file to hardware instructions, which will be deployed on FPGA.

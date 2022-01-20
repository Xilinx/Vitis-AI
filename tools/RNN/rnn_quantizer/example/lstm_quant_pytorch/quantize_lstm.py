
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataset, dataloader
#from torch.utils.data import BatchSampler

parser = argparse.ArgumentParser(description='Lstm pytorch quantizer test')
parser.add_argument('--quant_mode',
                    type=str, 
                    default='calib', 
                    help='Lstm pytorch quantization mode, calib for calibration of quantization, test for evaluation of quantized model')
parser.add_argument('--subset_len',
                    type=int,
                    default=None,
                    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
args = parser.parse_args()

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import tensorflow.keras as keras
from pytorch_nndct.apis import torch_quantizer

#torch.set_default_dtype(torch.double)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

top_words = 5000 # vocab size
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
#(X_train, y_train), (X_test, y_test) = imdb.load_data(path="./imdb.npz", num_words=top_words)

max_review_length = 500
X_test  = sequence.pad_sequences(X_test, maxlen=max_review_length)
test_data  = dataset.TensorDataset(torch.LongTensor(X_test), torch.Tensor(y_test))

if args.subset_len:
    subset_len = args.subset_len
    assert subset_len <= len(test_data)
    test_data = torch.utils.data.Subset(test_data, list(range(subset_len)))

#train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
test_loader  = dataloader.DataLoader(test_data, batch_size=50, shuffle=False)

class Model(nn.Module):
    def __init__(self, max_words, emb_size, hid_size):
        super(Model, self).__init__()
        self.max_words = max_words
        self.emb_size  = emb_size
        self.hid_size  = hid_size
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, batch_first=True) 
        self.fc = nn.Linear(self.hid_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.Embedding(x)
        x, _ = self.LSTM(x)
        x = x[:,-1,:]
        x = self.fc(x)
        out = self.sigmoid(x)
        return out.view(-1)

def test(model, device, test_loader):
    model.eval()
    model = model.cuda()
    criterion = nn.BCELoss()
    test_loss = 0.0 
    acc = 0 
    total_len = len(test_loader.dataset)
    print('---- Total test data size = {}'.format(total_len))
    for batch_idx, (x, y) in enumerate(test_loader):
        print('---- Testing batch {}'.format(batch_idx))
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        acc += (y.to(torch.int32) == (y_>0.5).to(torch.int32)).sum().item()
        #break
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)

embedding_vector_length = 32
hidden_vector_length = 100
model = Model(top_words, embedding_vector_length, hidden_vector_length).cpu()
model.load_state_dict(torch.load("./pretrained.pth"))

# nndct quantization
if args.quant_mode == 'calib' or args.quant_mode == 'test':
    netbak = model
    quantizer = torch_quantizer(quant_mode = args.quant_mode,
                                module = model,
                                bitwidth = 16,
                                lstm = True)
    model = quantizer.quant_model

# nndct quantization forwarding
acc = test(model, DEVICE, test_loader)
print("acc is: {:.4f}\n".format(acc))

# handle quantization result
if args.quant_mode == 'calib':
    quantizer.export_quant_config()
if args.quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=True)



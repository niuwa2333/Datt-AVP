# import
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
from time import sleep
import itertools
import numpy as np
from sklearn import metrics
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
##k-mer encoding
def encode_sequence(seq):
    k=3
    kmer_dict = {''.join(i): idx for idx, i in enumerate(itertools.product('ACDEFGHIKLMNPQRSTVWY', repeat=k))}
    encoding = torch.zeros(len(kmer_dict), dtype=torch.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        encoding[kmer_dict[kmer]] += 1

    return encoding
##load_file function demo: load excel data
def load_data(file, label):
  df = pd.read_excel(file)
  sequences = df.iloc[:, 1]
  features = []
  for seq in sequences:
    features.append(encode_sequence(seq))
  features = torch.stack(features).unsqueeze(1)
  labels = torch.tensor([label] * len(sequences), dtype=torch.long)
  return features, labels


####
#Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_out):
        out = self.linear(lstm_out)
        score = torch.bmm(out, out.transpose(1, 2))
        attn = self.softmax(score)

        context = torch.bmm(attn, lstm_out)
        return context
###
#LSTM-Att
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,drop):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # x shape (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out
##CNN-Att
class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,drop):
        super(CNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # x shape (batch, seq_len, input_size)
        out = x.permute(0, 2, 1)
        out = self.conv(out)
        out = out.permute(0, 2, 1)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out
##Datt-AVP
class DualModel(nn.Module):
    def __init__(self, input_size, hidden_size_cnn, hidden_size_lstm, num_layers_cnn, num_layers_lstm, num_classes,drop_cnn, drop_lstm):
        super(DualModel, self).__init__()
        self.cnn = CNNModel(input_size, hidden_size_cnn, num_layers_cnn, num_classes,drop_cnn)
        self.lstm = LSTMModel(input_size, hidden_size_lstm, num_layers_lstm, num_classes,drop_lstm)
        self.weight = nn.Parameter(torch.tensor(0.20523176251490569)) #learnable

    def forward(self, x):
        # x shape (batch, seq_len, input_size)
        out_cnn = self.cnn(x)
        out_lstm = self.lstm(x)
        out = self.weight * out_cnn + (1 - self.weight) * out_lstm
        out=torch.softmax(out,dim=1)####SOFTMAX
        return out

###
# best para for Datt-AVP
input_size = 8000  # k-mer table length
hidden_size_cnn=64
hidden_size_lstm = 65
num_layers_cnn=3
num_layers_lstm = 4
num_classes = 2
drop_cnn= 0.25196737781939776
drop_lstm=0.4261600226542729
L2= 0.020214585919074383
lr=2.8208083481812312e-05
model = DualModel(input_size, hidden_size_cnn, hidden_size_lstm, num_layers_cnn, num_layers_lstm, num_classes,drop_cnn, drop_lstm)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device('cuda')
print(torch.cuda.is_available())
model.to(device)

criterion = nn.CrossEntropyLoss()
# Adam
optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=L2)

##train
epochs = 122
for epoch in range(epochs):
    
    model.train()

    outputs = model(train_features)

    loss = criterion(outputs, train_labels)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
    train_preds = torch.argmax(outputs, dim=1)
    train_acc = torch.sum(train_preds == train_labels).item() / len(train_labels)
    model.eval()
    val_loss = criterion(val_outputs, test_labels).item()
    val_preds = torch.argmax(val_outputs, dim=1)
    val_acc = torch.sum(val_preds == test_labels).item() / len(test_labels)\
##eval on test set
model.eval()
outputs = model(test_set)
loss = criterion(outputs, test_lab)
test_loss = criterion(outputs, test_lab).item()
preds = torch.argmax(outputs, dim=1)

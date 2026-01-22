import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes = 1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        output = self.fc(last_output)
        return output 
    
class LSTM_RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes = 1):
        super(LSTM_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, (_, _) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output 
    
class GRU_RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes = 1):
        super(GRU_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.gru(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output 
    
class SOTA_RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, num_classes = 1, dropout_prob = 0.25):
        super(SOTA_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_prob if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, (_, _) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output_dropout = self.dropout(last_output)
        output = self.fc(last_output_dropout)
        return output 
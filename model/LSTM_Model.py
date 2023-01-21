import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(input_size=self.in_dim,  
                       hidden_size=self.hidden_dim,  
                       batch_first=True)  
        self.dense_1 = nn.Linear(hidden_dim, 1)
        self.avg_1d = nn.AvgPool1d(90)
        self.dense02 = nn.Linear(1+4,10)
        self.dense03 = nn.Linear(10,10)
        self.output = nn.Linear(10,n_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, pheno_input):
        self.lstm.flatten_parameters()
        fea,_= self.lstm(x)
        out_1_ = self.dense_1(fea)
        out_1 = torch.tanh(self.dropout(out_1_))
        out_1 = out_1.transpose(2, 1)
        pool = self.avg_1d(out_1)
        pool = pool.squeeze(1)
        merge = torch.cat([pool, pheno_input], dim=1)
        out_2 = torch.tanh(self.dense02(merge))
        out_2 = self.dropout(out_2)
        out_3 = self.dense03(out_2)
        outputs = self.sigmoid(self.output(out_3))
        return outputs,[fea, out_1_]


class Mutual_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Mutual_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(input_size=self.in_dim,  
                       hidden_size=self.hidden_dim,  
                       batch_first=True)  
        self.dense_1 = nn.Linear(hidden_dim, 1)
        self.avg_1d = nn.AvgPool1d(60)
        self.dense02 = nn.Linear(1+4,10)
        self.dense03 = nn.Linear(10,10)
        self.output = nn.Linear(10,n_classes)
        self.fc1_clu = nn.Linear(1+4, 10)
        self.fc2_clu = nn.Linear(10, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, pheno_input):
        self.lstm.flatten_parameters()
        fea,_= self.lstm(x)
        out_1_ = torch.tanh(self.dense_1(fea))
        out_1 = self.dropout(out_1_)
        out_1 = out_1.transpose(2, 1)
        pool = self.avg_1d(out_1)
        pool = pool.squeeze(1)
        merge = torch.cat([pool, pheno_input], dim=1)
        clu_1 = torch.tanh(self.fc1_clu(merge))
        clu_1 = self.dropout(clu_1)
        clu_2 = self.fc2_clu(clu_1)
        clu_outputs = self.sigmoid(clu_2)
        out_2 = torch.tanh(self.dense02(merge))
        out_2 = self.dropout(out_2)
        out_3 = self.dense03(out_2)
        outputs = self.sigmoid(self.output(out_3))
        return [outputs, clu_outputs],[fea, out_1_], out_2

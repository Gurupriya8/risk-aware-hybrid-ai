import torch
import torch.nn as nn

class SimpleMultimodalRiskModel(nn.Module):
    def __init__(self, input_dim, d_model=64, exog_in=0):
        super(SimpleMultimodalRiskModel, self).__init__()
        
        # LSTM backbone
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=d_model, batch_first=True)
        
        # Dropout + dense layer for classification
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, 1)  # binary classification

    def forward(self, x, exog=None):
        """
        x: (batch, features)
        We reshape it to (batch, seq_len=1, features) for LSTM
        """
        # Reshape: add a time dimension
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: (1, batch, d_model)
        
        # Take last hidden state
        h_last = h_n[-1]  # (batch, d_model)
        
        out = self.dropout(h_last)
        out = self.fc(out)  # (batch, 1)

        return out, h_last  # return logits + representation


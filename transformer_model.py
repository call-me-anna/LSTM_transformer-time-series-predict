import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerPredictor(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, seq_length=10):
        super(TransformerPredictor, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, 1)
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        predictions = self.decoder(transformer_out[:, -1, :])  # 只使用最后一个时间步的输出
        return predictions

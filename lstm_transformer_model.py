import torch
import torch.nn as nn

class LSTMTransformerPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, lstm_layers=1, 
                 d_model=64, nhead=4, transformer_layers=1, seq_length=10):
        super(LSTMTransformerPredictor, self).__init__()
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        
        # 维度转换
        self.transform = nn.Linear(hidden_size, d_model)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 输出层
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 维度转换
        transformed = self.transform(lstm_out)
        
        # Transformer处理
        transformer_out = self.transformer_encoder(transformed)
        
        # 最终预测
        predictions = self.decoder(transformer_out[:, -1, :])
        return predictions
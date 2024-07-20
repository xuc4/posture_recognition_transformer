import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Linear(2048, d_model)
        # Transform the output dimensions of ResNet to d_model

    def forward(self, src):
        src = self.embedding(src)
        src = src * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, num_queries=17, d_model=256, nhead=8, num_layers=6):
        super(TransformerDecoder, self).__init__()
        # Learnable query positional encoding
        self.query_pos = nn.Parameter(torch.rand(num_queries, d_model))
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        # Linear layer for predicting keypoints coordinates
        self.linear = nn.Linear(d_model, 2)

    def forward(self, memory):
        # Repeat query positional encoding to match the batch size
        tgt = self.query_pos.unsqueeze(1).repeat(1, memory.size(1), 1)
        output = self.transformer_decoder(tgt, memory)
        output = self.linear(output)
        return output

if __name__ == '__main__':
    # Test if the Transformer module is working correctly
    encoder = TransformerEncoder()
    decoder = TransformerDecoder()
    # Assume the input consists of 10 frames, each with 32 keypoint features, and each feature is 2048-dimensional
    input_tensor = torch.randn(10, 32, 2048)
    memory = encoder(input_tensor)
    output = decoder(memory)
    # It should output [17, 32, 2]
    print("Output shape:", output.shape)
from src.modeling.components import AbstractModelComponent
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Registers pe as a buffer that should not be considered a model parameter,
        # so it won't be updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is of shape (T, N, F)
        # pe is of shape (max_len, 1, F) and will be broadcasted to (T, N, F) during addition
        return x + self.pe[:x.size(0), :]

class CausalMultiheadAttentionBlock(nn.Module):
    def __init__(self, FRAME_EMBEDDING_SIZE, num_heads=8):
        super(CausalMultiheadAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.input_norm = nn.LayerNorm(FRAME_EMBEDDING_SIZE)
        self.attention = nn.MultiheadAttention(FRAME_EMBEDDING_SIZE, num_heads=num_heads)
        self.linear1 = nn.Linear(FRAME_EMBEDDING_SIZE, FRAME_EMBEDDING_SIZE)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_features=FRAME_EMBEDDING_SIZE, out_features=FRAME_EMBEDDING_SIZE)
        self.out_norm = nn.LayerNorm(FRAME_EMBEDDING_SIZE)

    def forward(self, x):
        x = self.input_norm(x)
        attn_mask = torch.triu(torch.ones(x.size(1)*self.num_heads,x.size(0), x.size(0)), diagonal=1).bool()
        x = self.attention(x,x,x,attn_mask=attn_mask,is_causal=True)[0]
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.out_norm(x)
        return x
class TemporalDecoder(AbstractModelComponent):
    """A block that takes in a batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, TEMPORAL_EMBEDDING_SIZE) and shape (N, T, TF), and outputs a batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, N_PHONEMES) and shape (N, T, C).
    """
    def __init__(self, FRAME_EMBEDDING_SIZE, N_PHONEMES, num_heads=8, attention_depth=18):
        super(TemporalDecoder, self).__init__()
        """Initialize the TemporalDecoder with a linear layer."""
        self.FRAME_EMBEDDING_SIZE = FRAME_EMBEDDING_SIZE
        self.N_PHONEMES = N_PHONEMES
        # compute num_heads from FRAME_EMBEDDING_SIZE
        # factor FRAME_EMBEDDING_SIZE into its prime factors
        self.attention_depth = attention_depth
        self.num_heads = num_heads
        if not FRAME_EMBEDDING_SIZE % self.num_heads == 0:
            raise ValueError(f"FRAME_EMBEDDING_SIZE ({FRAME_EMBEDDING_SIZE}) must be divisible by num_heads ({self.num_heads})")

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.positional_encoding = PositionalEncoding(FRAME_EMBEDDING_SIZE)
        attention_layers = []
        for _ in range(self.attention_depth):
            attention_layers.append(CausalMultiheadAttentionBlock(FRAME_EMBEDDING_SIZE, num_heads=self.num_heads))
        self.attention_layers = nn.ModuleList(attention_layers)

        self.fc = nn.Linear(FRAME_EMBEDDING_SIZE, N_PHONEMES)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        """Forward pass of the TemporalDecoder.
        """
        x = self.relu1(x)
        x = self.relu2(x)
        # generate a causal attention mask for the attention mechanism and a window size of 20
        # Attention mechanism: (N, T, TF) -> (T, N, TF)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        # (T, N, TF) -> (N, T, TF)
        x = x.transpose(0, 1)
        # Linear layer for classification: (N, T, TF) -> (N, T, C)
        x = self.fc(x)
        # Softmax to get probabilities: (N, T, C)
        x = self.softmax(x)
        return x

    @property
    def input_shape(self):
        return (self.FRAME_EMBEDDING_SIZE,)

    @property
    def output_shape(self):
        return (self.N_PHONEMES,)

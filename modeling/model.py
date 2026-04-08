import math
import torch
import torch.nn as nn


class LinearSeqModel(nn.Module):
    """A single linear layer mapping a flattened sequence to the next-row targets.

    Input dimension should be `seq_len * num_features` and output dimension is number
    of numeric target columns (6 for Open_change..log_market_cap).
    """

    def __init__(self, input_seq_length: int, latent_dim: int = 12, output_dim: int = 6):
        super().__init__()
        self.linear = nn.Linear(input_seq_length * output_dim, latent_dim)
        self.output = nn.Linear(latent_dim, output_dim)


    def forward(self, x):
        x = nn.functional.relu(self.linear(x.reshape(x.size(0), -1)))
        return torch.sigmoid(self.output(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # when d_model is odd, slice div_term to match
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe)  # shape: (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional encodings for input of shape (batch, seq_len, d_model).

        This returns a tensor shaped (1, seq_len, d_model) that can be added to inputs.
        """
        seq_len = x.size(1)
        return self.pe[:seq_len].unsqueeze(0)


class TransformerEncoderModel(nn.Module):
    """Encoder-only (BERT-like) transformer using a learned [CLS] token.

    The model expects input of shape `(batch, seq_len, num_features)` and returns
    a prediction for the next row based on the `CLS` token embedding.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.d_model = d_model

        # project input features to model dimension
        self.input_proj = nn.Linear(num_features, d_model)

        # learned [CLS] token (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # positional encoding supports seq_len + 1 (for CLS)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 1)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # simple head mapping CLS embedding to target dim
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (batch, seq_len, num_features)
        returns: (batch, output_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, num_features), got {x.shape}")

        batch_size, seq_len_in, _ = x.shape
        if seq_len_in > self.seq_len:
            raise ValueError(f"Input seq length {seq_len_in} exceeds model seq_len {self.seq_len}")

        # project features
        x_proj = self.input_proj(x)  # (batch, seq_len, d_model)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x_with_cls = torch.cat((cls_tokens, x_proj), dim=1)  # (batch, seq_len+1, d_model)

        # add positional encodings (ensure device/dtype match)
        pos = self.pos_encoder(x_with_cls).to(x_with_cls.dtype).to(x_with_cls.device)
        x_with_cls = x_with_cls + pos

        # transformer expects (seq_len, batch, d_model)
        x_t = x_with_cls.transpose(0, 1)
        encoded = self.encoder(x_t)  # (seq_len+1, batch, d_model)

        # CLS token is first position
        cls_emb = encoded[0]  # (batch, d_model)
        out = self.head(cls_emb)
        return out

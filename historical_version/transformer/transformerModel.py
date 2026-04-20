import torch
import torch.nn as nn
# import torchtune

from torchinfo import summary

class MarketTransformerEncoder(nn.Module):
    def __init__(self, num_stocks, model_dim=128, num_head=2, feedforward_dim=512, num_layer=3, batch_first=True, max_seq_length=256):
        super().__init__()
        self.embedding = nn.Linear(num_stocks, model_dim, bias=False)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_head, 
                dim_feedforward=feedforward_dim, 
                batch_first=batch_first
            ),
            num_layers=num_layer
        )
        # self.rope = torchtune.modules.RotaryPositionalEmbeddings(dim=model_dim)
        self.max_seq_length = max_seq_length
        self.final_projector = nn.Linear(model_dim, num_stocks)
    
    def forward(self, input):
        if input.shape[1] > self.max_seq_length:
            raise NotImplementedError(f"Sequence length exceeded the maximum sequence length: {self.max_seq_length}")
        x = self.embedding(input)
        x = self.encoder(x)
        x = self.final_projector(x) * input
        return x
    
if __name__ == "__main__":
    encoder = MarketTransformerEncoder()
    summary(encoder)
    data = torch.rand((4, 256, 16))
    print(encoder(data))
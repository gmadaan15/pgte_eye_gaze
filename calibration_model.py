import torch
from torchvision import models
from torch import nn

# model for calibration model
class CalibrationModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_units, num_heads, num_encoder_layers, num_queries):
        super().__init__()
        self.num_queries = num_queries
        # mlp, encoder, mlp
        self.initial_mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU()
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(in_features=hidden_units, out_features=out_features),
            nn.ReLU()
        )

    def forward(self, x):
        #x = self.pass_to_initialMLP(x)
        x = self.initial_mlp(x)
        x = self.encoder(x)
        x = self.final_mlp(x)
        # finally lets take average of the encodings
        return torch.mean(x, dim=1)

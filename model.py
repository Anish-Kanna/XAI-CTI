import torch
import torch.nn as nn

class XAI_CTI_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=2,
                batch_first=True
            ),
            num_layers=1
        )

        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1)          # [batch, 1, features]
        x = self.cnn(x)             # [batch, 32, features/2]
        x = x.permute(0, 2, 1)      # [batch, seq, 32]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)
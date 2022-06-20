import torch
import torch.nn as nn


class BSENTClassificationH(nn.Module):
    def __init__(self, device, model):
        super(BSENTClassificationH, self).__init__()
        self.device = device
        self.model = model
        self.fc_layer = nn.Sequential(
            nn.Linear(6768, 4096),
            # nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 7),
            nn.Softmax(dim=-1)
        )

    def forward(self, x1, attention_mask, mfcc):
        # classification with cls token
        x1_emb = self.model(x1, attention_mask=attention_mask.float().to(self.device)).pooler_output.to(self.device)
        x = torch.cat((x1_emb, mfcc.to(self.device)), dim=1).to(self.device)
        x = self.fc_layer(x)
        return x

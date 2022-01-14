import torch
import torch.nn as nn


class NeuralNetworkRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_layer = nn.Linear(7, 7)
        self.act_emb = nn.Tanh()
        self.layer1 = nn.Linear(7 + 7, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.act_1 = nn.LeakyReLU()
        self.d1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(100, 40)
        self.bn2 = nn.BatchNorm1d(40)
        self.act_2 = nn.LeakyReLU()
        self.d2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(40, 1)

    def forward(self, x, cat_x):
        cat_x_embedded = self.emb_layer(cat_x)
        cat_x_embedded = self.act_emb(cat_x_embedded)
        x = torch.cat([x,cat_x_embedded],dim=1)
        activation1 = self.act_1(self.bn1(self.layer1(x)))
        activation1 = self.d1(activation1)
        activation2 = self.act_2(self.layer2(activation1))
        activation2 = self.d1(activation2)
        output = self.layer3(activation2)
        return output
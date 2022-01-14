import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_layer = nn.Linear(7, 7)
        self.act_emb = nn.Tanh()
        self.layer1 = nn.Linear(7 + 7, 100)
        self.act_1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(100, 40)
        self.act_2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(40, 1)

    def forward(self, x, cat_x):
        cat_x_embedded = self.emb_layer(cat_x)
        cat_x_embedded = self.act_emb(cat_x_embedded)
        x = torch.cat([x,cat_x_embedded],dim=1)
        activation1 = self.act_1(self.layer1(x))
        activation2 = self.act_2(self.layer2(activation1))
        output = self.layer3(activation2)
        return output
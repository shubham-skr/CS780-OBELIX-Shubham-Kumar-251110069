
import os
import torch
import torch.nn as nn
import numpy as np

ACTIONS = ["L45","L22","FW","R22","R45"]

class DuelingNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(18, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 5)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.adv(x)
        return v + (a - a.mean(dim=1, keepdim=True))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights.pth")
model = DuelingNet()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

def policy(obs, rng):
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q = model(x)
        action = torch.argmax(q).item()
    return ACTIONS[action]
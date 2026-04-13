import numpy as np
import torch
import torch.nn as nn
import os
from collections import deque
import random

# actions
A = ["L45","L22","FW","R22","R45"]

def make_state(o, att, last, sc):
    r = np.sum(o[0:4])
    f = np.sum(o[4:12])
    l = np.sum(o[12:16])
    ir, st = o[16], o[17]
    # simplified flags
    wall = int(st == 1 and (l+f+r) > 0)
    bound = int(st == 1 and (l+f+r) == 0)
    return np.array([l,f,r,ir,st,att,wall,bound,last,sc], dtype=np.float32)


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.pi = nn.Linear(128, 5)
        self.v  = nn.Linear(128, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.pi(h), self.v(h)

net = None
mem = deque(maxlen=4)

att, last, prev = 0, 2, 2
sc, infrared = 0, 0

stk_m = 0
stk_d = 0
rl_calls = 0

def load():
    global net
    if net is not None: return

    net = Net(40)
    base = os.path.dirname(os.path.abspath(__file__))
    pth = os.path.join(base, "weights.pth")
    
    sd = torch.load(pth, map_location="cpu")
    net.load_state_dict(sd)
    net.eval()


def rule(o):
    global att, last, prev, sc, stk_m, stk_d, infrared
    r = np.sum(o[0:4])
    f = np.sum(o[4:12])
    l = np.sum(o[12:16])
    ir, st = o[16], o[17]
    if ir == 1:
        infrared += 1
    elif st == 1: 
        infrared = 0
        att = 0
    else:
        infrared = 0

    if st == 0 and ir == 1 and f == 0 and l == 0 and r == 0 and infrared >= 4:
        att = 1

    if st == 1:
        if stk_m == 0:
            stk_d = random.choice([0,4]) 
            stk_m = 1
            return stk_d
        elif stk_m == 1:
            stk_m = 2
            return stk_d
        else:
            stk_m = 0
            return 2 
    else:
        stk_m = 0

    if att == 1:
        last = 2
        return 2
    
    if att == 0 and last == 2:
        if not ((l > 0 and r > 0)):
            if l > 0 and r == 0:
                last = 0
                return 0 
            if r > 0 and l == 0:
                last = 4
                return 4 
            if f > 0:
                last = 2
                return 2 
            
    return None


@torch.no_grad()
def policy(o, rng):
    global mem, att, last, sc, rl_calls
    load()
    
    ra = rule(o)
    s = make_state(o, att, last, sc)

    if len(mem) == 0:
        for _ in range(4): mem.append(s)

    mem.append(s)

    if ra is not None:
        last = ra
        return A[ra]

    rl_calls += 1
    
    flat = np.concatenate(mem)
    inp = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

    logits, _ = net(inp)
    idx = int(torch.argmax(logits, dim=1).item())

    last = idx
    return A[idx]

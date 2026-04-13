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

# global vars for tracking
net = None
mem = deque(maxlen=4)

att, last, prev = 0, 2, 2
sc, infrared = 0, 0

# stuck recovery stuff
stk_m = 0
stk_d = 0

def load():
    global net
    if net is not None: return

    net = Net(40)
    # get path relative to this file
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
    else:
        infrared = 0
    
    print(att)
    # check if we should attach
    if st == 0 and ir == 1 and f == 0 and l == 0 and r == 0 and infrared >= 4:
        att = 1

    # --- RECOVERY LOGIC ---
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
            return 2 # push fwd
    else:
        stk_m = 0
    
    # --- PUSHING ---
    if att == 1:
        last = 2
        return 2
    
    # --- SEEKING ---
    if att == 0:
        if last != 2:
            last = 2
            return 2
        
        if not (l > 0 and r > 0):
            
            if l > 0 and r == 0:
                last = 0
                return 0 # L22

            if r > 0 and l == 0:
                last = 4
                return 4 # R22

            if f > 0:
                last = 2
                return 2 
            
            else:
                # just keep moving
                last = 2
                return 2

    # --- EXPLORE / CORRIDORS ---
    if np.sum(o[:16]) == 0 and att == 0:
        return 2 

    if l > 0 and r > 0 and f == 0:
        return 2

    return None


@torch.no_grad()
def policy(o, rng):
    global mem, att, last, sc

    load()
    ra = rule(o)
    s = make_state(o, att, last, sc)

    # init buffer if empty
    if len(mem) == 0:
        for _ in range(4): mem.append(s)

    mem.append(s)

    # rule based override
    if ra is not None:
        return A[ra]

    # model inference
    flat = np.concatenate(mem)
    inp = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

    logits, _ = net(inp)
    idx = int(torch.argmax(logits, dim=1).item())

    last = idx
    return A[idx]
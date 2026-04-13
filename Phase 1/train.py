# D3QN-PER

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from obelix import OBELIX

A = ["L45","L22","FW","R22","R45"]

def greedy(net, s):
    s = torch.FloatTensor(s).unsqueeze(0)
    with torch.no_grad():
        q = net(s)
    return torch.argmax(q).item()

def eps_greedy(net, s, eps):
    if random.random() < eps:
        return random.randint(0,4)
    return greedy(net, s)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(18,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.v = nn.Linear(256,1)
        self.a = nn.Linear(256,5)

    def forward(self,x):
        x = self.f(x)
        v = self.v(x)
        a = self.a(x)
        return v + (a - a.mean(dim=1, keepdim=True))

class Buf:
    def __init__(self, n):
        self.n = n
        self.mem = []
        self.i = 0
        self.p = np.zeros(n)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_inc = 0.001
        self.eps = 1e-6

    def add(self, exp):
        mx = self.p.max() if len(self.mem)>0 else 1.0
        if len(self.mem) < self.n:
            self.mem.append(exp)
        else:
            self.mem[self.i] = exp
        self.p[self.i] = mx
        self.i = (self.i + 1) % self.n

    def sample(self, bs):
        sz = len(self.mem)
        pr = self.p[:sz] ** self.alpha
        pr /= pr.sum()
        idx = np.random.choice(sz, bs, p=pr)
        samp = [self.mem[k] for k in idx]
        w = (sz * pr[idx]) ** (-self.beta)
        w /= w.max()
        self.beta = min(1.0, self.beta + self.beta_inc)

        return samp, idx, w

    def update(self, idx, err):
        for i,e in zip(idx, err):
            self.p[i] = abs(e) + self.eps

    def size(self):
        return len(self.mem)

class Agent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.tau = 0.01
        self.bs = 128
        self.buf = Buf(50000)
        self.eps = 1.0
        self.net = Net()
        self.tgt = Net()
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

    def soft_update(self):
        for t, s in zip(self.tgt.parameters(), self.net.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def step(self):
        samp, idx, w = self.buf.sample(self.bs)
        s,a,r,ns,d = zip(*samp)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        ns = torch.FloatTensor(ns)
        d = torch.FloatTensor(d)
        w = torch.FloatTensor(w)
        q = self.net(s).gather(1, a.unsqueeze(1)).squeeze()
        na = self.net(ns).argmax(1)
        nq = self.tgt(ns).gather(1, na.unsqueeze(1)).squeeze()
        tgt = r + self.gamma * nq * (1 - d)
        err = tgt.detach() - q
        loss = (w * err.pow(2)).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.buf.update(idx, err.detach().numpy())

    def train(self, eps=3000):
        for ep in range(eps):
            s = self.env.reset()
            done = False
            tot = 0
            while not done:
                a = eps_greedy(self.net, s, self.eps)
                ns, r, done = self.env.step(A[a], render=False)
                self.buf.add((s, a, r, ns, float(done)))
                s = ns
                tot += r
                if self.buf.size() > self.bs:
                    self.step()
                    self.soft_update()
            if ep < 2000:
                self.eps = 1.0 - 0.0004 * ep
            else:
                self.eps = 0.05
            print("ep", ep, "r", tot, "eps", self.eps)
        torch.save(self.net.state_dict(), "weights.pth")


if __name__ == "__main__":
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=1000,
        wall_obstacles=True,
        difficulty=0
    )
    ag = Agent(env)
    ag.train(5000)
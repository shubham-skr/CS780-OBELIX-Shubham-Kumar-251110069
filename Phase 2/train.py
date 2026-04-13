# PPO
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from obelix import OBELIX
import random

A = ["L45","L22","FW","R22","R45"]

def make_state(o, att, last, sc):
    r = np.sum(o[0:4])
    f = np.sum(o[4:12])
    l = np.sum(o[12:16])
    ir = o[16]
    st = o[17]
    wall = 1 if (st == 1 and (l+f+r) > 0) else 0
    bound = 1 if (st == 1 and (l+f+r) == 0) else 0
    return np.array([l,f,r,ir,st,att,wall,bound,last,sc], dtype=np.float32)

class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )
        self.pi = nn.Linear(128,5)
        self.v  = nn.Linear(128,1)

    def forward(self,x):
        h = self.fc(x)
        return self.pi(h), self.v(h)

class Ctrl:
    def __init__(self):
        self.att = 0
        self.last = 2
        self.prev = 2
        self.sc = 0
        self.si = 0
        self.search = ["FW","FW","FW","FW","FW","L22","L22","L22","L22"]

    def reset(self):
        self.att = 0
        self.last = 2
        self.prev = 2
        self.sc = 0
        self.si = 0

    def act(self, o):
        r = np.sum(o[0:4])
        f = np.sum(o[4:12])
        l = np.sum(o[12:16])
        ir = o[16]
        st = o[17]
        if st == 0 and ir == 1 and f == 0:
            self.att = 1

        if st == 1:
            self.sc += 1

            if self.prev == self.last and self.last in [0,4]:
                self.last = 2
                return 2

            self.prev = self.last
            self.last = random.choice([0,4])
            return self.last
        else:
            self.sc = 0

        if self.att == 1:
            self.prev = self.last
            self.last = 2
            return 2

        if np.sum(o[:16]) == 0:
            a = self.search[self.si % len(self.search)]
            self.si += 1
            self.prev = self.last
            self.last = A.index(a)
            return self.last

        if l > 0 and r > 0 and f == 0:
            self.prev = self.last
            self.last = 2
            return 2

        return None

class PPO:
    def __init__(self, env):
        self.env = env
        self.ctrl = Ctrl()
        self.stack = 4
        self.buf = deque(maxlen=self.stack)
        self.sdim = 10 * self.stack
        self.net = Net(self.sdim)
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.clip = 0.2

    def get_state(self, o):
        s = make_state(o, self.ctrl.att, self.ctrl.last, self.ctrl.sc)
        if len(self.buf) == 0:
            for _ in range(self.stack):
                self.buf.append(s)
        self.buf.append(s)
        return np.concatenate(self.buf)

    def act(self, s):
        x = torch.FloatTensor(s).unsqueeze(0)
        logits, v = self.net(x)
        p = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(p)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a.item(), logp.detach(), v.squeeze().detach()

    def train(self):
        for ep in range(2000):
            o = self.env.reset()
            self.ctrl.reset()
            self.buf.clear()
            traj = []
            done = False
            tot = 0
            while not done:
                ra = self.ctrl.act(o)
                s = self.get_state(o)
                if ra is not None:
                    a = ra
                    logp = torch.tensor(0.0)
                    v = torch.tensor(0.0)
                else:
                    a, logp, v = self.act(s)

                no, r, done = self.env.step(A[a], render=False)
                traj.append((s,a,logp,v,r,done))
                o = no
                tot += r

            self.update(traj)
            print("ep", ep, "r", round(tot,1))

        torch.save(self.net.state_dict(), "ppo_final.pth")

    def update(self, traj):
        s,a,lp,v,r,d = zip(*traj)
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a)
        old_lp = torch.tensor(lp, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)
        ret = []
        G = 0
        for rr,dd in reversed(list(zip(r,d))):
            G = rr + self.gamma * G * (1-dd)
            ret.insert(0,G)

        ret = torch.FloatTensor(ret)
        adv = ret - v
        for _ in range(4):
            logits, vv = self.net(s)
            p = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(p)
            new_lp = dist.log_prob(a)
            ratio = torch.exp(new_lp - old_lp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv
            loss = -torch.min(s1,s2).mean() + 0.5*F.mse_loss(vv.squeeze(), ret)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

if __name__ == "__main__":
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=2000,
        wall_obstacles=True,
        difficulty=2
    )
    ag = PPO(env)
    ag.train()
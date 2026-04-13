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
    return np.array([l, f, r, ir, st, att, wall, bound, last, sc], dtype=np.float32)


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, 256),
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


class Ctrl:
    def __init__(self):
        self.att = 0
        self.last = 2
        self.prev = 2
        self.sc = 0
        self.search_i = 0
        self.search = ["FW","FW","FW","FW","FW","L22","L22","L22","L22"]

    def reset(self):
        self.att = 0
        self.last = 2
        self.prev = 2
        self.sc = 0
        self.search_i = 0

    def act_rule(self, o):
        r = np.sum(o[0:4])
        f = np.sum(o[4:12])
        l = np.sum(o[12:16])
        ir = o[16]
        st = o[17]
        # attach guess
        if st == 0 and ir == 1 and f == 0:
            self.att = 1
        # stuck case
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
        # pushing
        if self.att == 1:
            self.prev = self.last
            self.last = 2
            return 2
        # no signal -> explore
        if np.sum(o[:16]) == 0:
            act = self.search[self.search_i % len(self.search)]
            self.search_i += 1
            self.prev = self.last
            self.last = A.index(act)
            return self.last
        # corridor case
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
        self.state_dim = 10 * self.stack
        self.net = Net(self.state_dim)
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

    def select(self, s):
        x = torch.FloatTensor(s).unsqueeze(0)
        logits, v = self.net(x)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
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
            total = 0
            while not done:
                ra = self.ctrl.act_rule(o)
                s = self.get_state(o)
                if ra is not None:
                    a = ra
                    logp = torch.tensor(0.0)
                    v = torch.tensor(0.0)
                else:
                    a, logp, v = self.select(s)

                no, r, done = self.env.step(A[a], render=False)
                # small shaping
                if no[16] == 1:
                    r += 2
                if np.sum(no[4:12]) > 0:
                    r += 1
                traj.append((s, a, logp, v, r, done))
                o = no
                total += r

            self.update(traj)
            print("Ep", ep, "| R", round(total,1))
        torch.save(self.net.state_dict(), "ppo_final.pth")


    def update(self, traj):
        states, actions, logps, vals, rewards, dones = zip(*traj)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_logps = torch.tensor(logps, dtype=torch.float32)
        vals = torch.tensor(vals, dtype=torch.float32)
        returns = []
        G = 0
        for r, d in reversed(list(zip(rewards, dones))):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        adv = returns - vals
        for _ in range(4):
            logits, v = self.net(states)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            new_logps = dist.log_prob(actions)
            ratio = torch.exp(new_logps - old_logps)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv
            loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(v.squeeze(), returns)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


if __name__ == "__main__":
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=2000,
        wall_obstacles=True,
        difficulty=3
    )
    agent = PPO(env)
    agent.train()
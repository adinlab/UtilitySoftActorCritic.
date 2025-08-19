import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, arch, n_state, n_action):
        super(Actor, self).__init__()
        self.n_hidden = 256
        self.device = "cpu"
        self.learning_rate = 3e-4
        self.model = arch(n_state, n_action, self.n_hidden).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def act(self, s, is_training=True):
        a, e = self.model(s, is_training=is_training)
        return a, e

    def loss(self, s, a, e, critics, alpha):
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)
        return (-q + alpha * e).mean()

    def update(self, s, critics, alpha):
        self.optim.zero_grad()
        a, e = self.act(s)
        loss = self.loss(s, a, e, critics, alpha)
        loss.backward()
        self.optim.step()
        return a, e


class Critic(nn.Module):
    def __init__(self, arch, n_state, n_action):
        super(Critic, self).__init__()
        self.n_hidden = 256
        self.device = "cpu"
        self.learning_rate = 3e-4
        self.tau = 0.005
        self.model = arch(n_state, n_action, self.n_hidden).to(self.device)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.target = arch(n_state, n_action, self.n_hidden).to(self.device)
        self.init_target()

    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def Q(self, s, a):
        return self.model(s, a)

    def Q_t(self, s, a):
        return self.target(s, a)

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), y)
        loss.backward()
        self.optim.step()


class CriticEnsemble(nn.Module):
    def __init__(self, arch, n_state, n_action, critictype=Critic):
        super(CriticEnsemble, self).__init__()
        self.n_elements = 2
        self.critics = [
            critictype(arch, n_state, n_action) for _ in range(self.n_elements)
        ]

    def __getitem__(self, item):
        return self.critics[item]

    def Q(self, s, a):
        return [critic.Q(s, a) for critic in self.critics]

    def Q_t(self, s, a):
        return [critic.Q_t(s, a) for critic in self.critics]

    def update(self, s, a, y):
        [critic.update(s, a, y) for critic in self.critics]

    def update_target(self):
        [critic.update_target() for critic in self.critics]

    def reduce(self, q_val_list):
        # Reduces the outputs of ensemble elements into a single value
        return torch.cat(q_val_list, dim=-1).min(dim=-1, keepdim=True)[0]

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor, alpha):
        ap, ep = actor.act(sp)
        qp = self.Q_t(sp, ap)
        qp_t = self.reduce(qp) - alpha * ep
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y

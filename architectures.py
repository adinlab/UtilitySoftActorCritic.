
import torch,math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

class CriticNet(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256):
        super(CriticNet, self).__init__()
        self.arch = nn.Sequential(
            nn.Linear(n_x[0] + n_u[0], n_hidden),
            nn.ReLU(),
            ###
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            ###
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        return self.arch(f)
 
   
class SquashedGaussianHead(nn.Module):
    def __init__(self, n, upper_clamp=-2.0):
        super(SquashedGaussianHead, self).__init__()
        self._n = n
        self._upper_clamp = upper_clamp

    def forward(self, x, is_training=True):
        mean_bt = x[..., : self._n]
        log_var_bt = (x[..., self._n :]).clamp(
            -13, None
        )  
        std_bt = log_var_bt.exp().sqrt()
        dist_bt = Normal(mean_bt, std_bt)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist_bt, transform)
        if is_training:
            y = dist.rsample()
            y_logprob = dist.log_prob(y).sum(dim=-1, keepdim=True)
        else:
            y_samples = dist.rsample((100,))
            y = y_samples.mean(dim=0)
            y_logprob = None

        return y, y_logprob  

class ActorNetProbabilistic(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256, upper_clamp=-2.0):
        super(ActorNetProbabilistic, self).__init__()
        self.n_u = n_u
        self.arch = nn.Sequential(
            nn.Linear(n_x[0], n_hidden),
            nn.ReLU(),
            ##
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            ##
            nn.Linear(n_hidden, 2 * n_u[0]),
        )
        self.head = SquashedGaussianHead(self.n_u[0], upper_clamp)

    def forward(self, x, is_training=True):
        f = self.arch(x)
        return self.head(f, is_training)

import torch
from torch import nn
from sac import  SoftActorCritic
from models_basic import Actor,Critic,CriticEnsemble
import math

class UtilityCriticEnsemble(CriticEnsemble):

    def g(self, kappa):
        try:
            result = math.log(1 / (1 - kappa**2)) / (math.sqrt(2) * kappa)
        except ZeroDivisionError:
            result = 0
        return result

    def get_utility(self, s, a, kappa, is_target=False):
        if is_target:
            q_list = self.Q_t(s, a)
        else:
            q_list = self.Q(s, a)

        q_cat = torch.cat(q_list, dim=-1)
        
        q_mean = torch.mean(q_cat, dim=1, keepdim=True)
        q_std = torch.std(q_cat, dim=1, keepdim=True, correction=0)
        q_u = q_mean + self.g(kappa) * q_std
    
        return q_u

    def U(self, s, a, kappa):  # utility of the ensemble    
        return self.get_utility(s, a, kappa, is_target=False)
    
    def U_t(self, s, a, kappa):
        return self.get_utility(s, a, kappa, is_target=True)
    
    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor, alpha):
        self.kappa_critic = -(0.831559)
        self.gamma = 0.99
        ap, ep = actor.act(sp)
        qp_t =  self.U_t(sp, ap, self.kappa_critic) - alpha * ep
        return r.unsqueeze(-1) + (self.gamma  * qp_t * (1 - done.unsqueeze(-1)))
    
class UtilityActor(Actor):
    def __init__(self, arch,  n_state, n_action):
        super(UtilityActor, self).__init__(arch,  n_state, n_action)
        self.iter = 0
        self.max_steps = 100000
        self.kappa_actor = -(0.831559)
        self.max_iter = self.max_steps
        self.kappa = self.kappa_actor

    def loss(self, s, a, e, critics, alpha):
        q = critics.U(s, a, self.kappa)
        return (-q + alpha * e).mean()

class UtilitySoftActorCritic(SoftActorCritic):
    _agent_name = "USAC"

    def __init__(self, env, actor_nn, critic_nn):
        super(UtilitySoftActorCritic, self).__init__(env,  actor_nn, critic_nn, UtilityCriticEnsemble, UtilityActor) 
        self.kappa_actor = -(0.831559)
        self.kappa_critic = -(0.831559)
        self.max_steps = 100000
        self.gamma = 0.99

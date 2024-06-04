import math,torch
from agent import Agent
from models_basic import Actor,Critic, CriticEnsemble

class SoftActorCritic(Agent):
    _agent_name = "SAC"

    def __init__(
        self,
        env,
        actor_nn,
        critic_nn,
        CriticEnsembleType=CriticEnsemble,
        ActorType=Actor,
        
    ):
        super(SoftActorCritic, self).__init__(env)
        self.critics = CriticEnsembleType(critic_nn, self._nx, self._nu)
        self.actor = ActorType(actor_nn,  self._nx, self._nu)

        self.alpha=0.2
        self.n_hidden = 256
        self.learning_rate = 3e-4
        self._step = 0
        self._max_steps = 100000
        self.batch_size=256
        self._gamma=0.99

        self.H_target = -self._nu[0]
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], self.learning_rate)
        self.log_alpha.data.fill_(math.log(self.alpha))

     
    def update_alpha(self, e):
        alpha_loss = -(self.log_alpha.exp() * (e + self.H_target).detach()).mean()
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()

    def learn(self, max_iter=1):
        if self.batch_size > len(self.experience_memory):
            return None

        for iteration in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.batch_size
            )
            y = self.critics.get_bellman_target(
                r, sp, done, self.actor, self.log_alpha.exp()
            )
            self.critics.update(s, a, y)
            a, e = self.actor.update(s, self.critics, self.log_alpha.exp())
            self.update_alpha(e)
            self.critics.update_target()

    @torch.no_grad()
    def select_action(self, s, is_training=True):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        a, _ = self.actor.act(s, is_training=is_training)
        a = a.cpu().numpy().squeeze(0)
        return a

    def Q_value(self, s, a):
        s = torch.from_numpy(s).view(1, -1).float().to(self.device)
        a = torch.from_numpy(a).view(1, -1).float().to(self.device)
        q = self.critics[0].Q(s, a)
        return q.item()

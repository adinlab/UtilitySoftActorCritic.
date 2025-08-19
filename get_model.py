from architectures import CriticNet, ActorNetProbabilistic
from usac import UtilitySoftActorCritic


def get_model(env):
    model_name = "USAC"
    critic_nn = CriticNet
    actor_nn = ActorNetProbabilistic
    return UtilitySoftActorCritic(env, actor_nn, critic_nn)

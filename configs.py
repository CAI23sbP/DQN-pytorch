import torch.nn as nn
import gym 
import torch.optim as optim
from gym.wrappers.normalize import *

class BaseConfig():
    def __init__(self):
        pass

class Config():

    Env = BaseConfig()
    Env.name = "CartPole-v1"
    Env.norm_reward = False
    Env.norm_obs = False
    Env.device = "cpu"
    Env.training_render = True
    Env.testing_render = True
    Env.num_episodes = 600 ## for training
    Env.max_step = 500 ## for training
    Env.save_file_name = "dqn_model"

    Env.te_num_episodes = 10 ## for testing
    Env.te_max_step = 500 ##for testing

    Env.make_env = gym.make(Env.name)
    if Env.norm_obs:
        Env.make_env = NormalizeObservation(Env.make_env)
    if Env.norm_reward:
        Env.make_env = NormalizeReward(Env.make_env)

    Env.action_space = Env.make_env.action_space 
    Env.observation_space = Env.make_env.observation_space.shape[0] 


    Buffer = BaseConfig()
    Buffer.capacity = 10000
    Buffer.BATCH_SIZE = 128

    Buffer.tuple = {"Transition":("state","action","next_state","reward","done")}

    Network = BaseConfig()
    Network.mlp_dims = [128,128,128,Env.action_space.n]
    Network.activation = nn.ReLU()
    Network.optimizer = optim.AdamW
    Network.loss_function = nn.MSELoss()
    Network.LR = 3e-1
    Network.GAMMA = 0.99
    Network.EPS_START = 0.9
    Network.EPS_END = 5e-2
    Network.EPS_DECAY = 1e3
    Network.TAU = 5e-3

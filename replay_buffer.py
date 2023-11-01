
import random,torch
from collections import namedtuple, deque


class ReplayBuffer():
    def __init__(self, configs, predict_net, target_net, opt_lossf):
        self.config = configs
        self.memory = deque([], maxlen= self.config.Buffer.capacity)
        key = list(self.config.Buffer.tuple.keys())[0]
        self.Transition = namedtuple(key,
                        self.config.Buffer.tuple[f"{key}"])
        self.init_batch()
        self.predict_net = predict_net
        self.target_net = target_net 
        self.optimizer, self.criterion = opt_lossf

    def add(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.config.Buffer.BATCH_SIZE)

    def __len__(self):
        return len(self.memory)

    def init_batch(self):
        self.state_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,self.config.Env.observation_space)
        self.action_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,self.config.Env.action_space.n)
        self.next_state_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,self.config.Env.observation_space)
        self.reward_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,1)
        self.done_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,1,dtype = torch.bool)
        self.next_value = torch.zeros(self.config.Buffer.BATCH_SIZE, device=self.config.Env.device)

    def update(self,*args):
        self.add(*args)
        if len(self.memory) < self.config.Buffer.BATCH_SIZE:
            return
        dataset = self.sample()
        
        for i, data in enumerate([*dataset]):
            self.state_batch[i] = data[0]
            self.action_batch [i] = data[1] 
            self.next_state_batch[i] = data[2]
            self.reward_batch [i] = data[3]
            self.done_batch [i] = data[4] 

        predict_q_value = self.predict_net(self.state_batch)

        with torch.no_grad():
            next_states = torch.cat([next_state for done, next_state in zip(self.done_batch, self.next_state_batch) if not done]).reshape(-1,4)
            self.next_value[~self.done_batch.view(-1)] = self.target_net(next_states).max(1)[0]
        
        expected_func = predict_q_value * self.config.Network.GAMMA + self.reward_batch
        loss = self.criterion(predict_q_value, expected_func.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

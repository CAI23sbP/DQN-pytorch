
import random,torch, os
from collections import namedtuple, deque
from torch.utils.tensorboard import SummaryWriter

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
        self.writer = SummaryWriter(os.getcwd())
        self.iter = 0

    def add(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.config.Buffer.BATCH_SIZE)

    def __len__(self):
        return len(self.memory)

    def init_batch(self):
        self.state_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,self.config.Env.observation_space)
        self.action_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,1, dtype=torch.int64)
        self.next_state_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,self.config.Env.observation_space)
        self.reward_batch = torch.zeros(self.config.Buffer.BATCH_SIZE)
        self.done_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,1,dtype = torch.bool)
        self.next_value = torch.zeros(self.config.Buffer.BATCH_SIZE, device=self.config.Env.device)

    def on_step(self):
        self.iter +=1
        self.iter = int(self.iter/ 10)

    def update(self,*args):
        self.add(*args)
        self.on_step()
        if len(self.memory) < self.config.Buffer.BATCH_SIZE:
            return
        dataset = self.sample()
        for i, data in enumerate([*dataset]):
            self.state_batch[i] = data[0]
            self.action_batch [i] = data[1] 
            self.next_state_batch[i] = data[2]
            self.reward_batch [i] = data[3]
            self.done_batch [i] = data[4] 

        predict_q_value = self.predict_net(self.state_batch).gather(1, self.action_batch)

        with torch.no_grad():
            next_states = torch.cat([next_state for done, next_state in zip(self.done_batch, self.next_state_batch) if not done]).reshape(-1,self.config.Env.observation_space)
            self.next_value[~self.done_batch.view(-1)] = self.target_net(next_states).max(1)[0]
            
            rewards = self.reward_batch[~self.done_batch.view(-1)]
            G_return = 0
            for i in range(len(rewards)-1):
                G_return += rewards[i]*self.config.Network.GAMMA 
            G_return+=rewards[-1].item()
        expected_func = self.next_value * self.config.Network.GAMMA + self.reward_batch

        loss = self.criterion(predict_q_value, expected_func)
        tag_scalar_loss= {"Q_value_loss": loss}
        tag_scalar_return= {"return": G_return}
        self.writer.add_scalars("DQN",tag_scalar_return, self.iter)
        self.writer.add_scalars("DQN",tag_scalar_loss, self.iter)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.predict_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config.Network.TAU + target_net_state_dict[key]*(1- self.config.Network.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self):
        path = os.path.join(os.getcwd(),f"{self.config.Env.save_file_name}.pt")
        print(f"save path {path}")
        torch.save(self.target_net.state_dict(), path)
    
    def load(self):
        path = os.path.join(os.getcwd(),f"{self.config.Env.save_file_name}.pt")
        print(f"load path {path}")
        self.target_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.predict_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

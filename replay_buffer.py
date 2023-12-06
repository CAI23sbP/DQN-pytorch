
import random,torch, os
from collections import namedtuple, deque

class DQNBuffer():
    def __init__(self, configs, predict_net, target_net, opt_lossf):
        self.config = configs
        self.memory = deque([], maxlen= self.config.Buffer.capacity)
        tuples = {"Transition":("state","action","next_state","reward","done")}
        key = list(tuples.keys())[0]
        self.Transition = namedtuple(key,
                        tuples[f"{key}"])
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
        self.action_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,1, dtype=torch.int64)
        self.next_state_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,self.config.Env.observation_space)
        self.reward_batch = torch.zeros(self.config.Buffer.BATCH_SIZE)
        self.done_batch = torch.zeros(self.config.Buffer.BATCH_SIZE,dtype = torch.bool)
        
    def update(self,*args):
        self.add(*args)
        #state, action, next_state, reward, done
        if len(self.memory) < self.config.Buffer.BATCH_SIZE:
            return
        
        for n in range(self.config.Network.epoch):
            dataset = self.sample()
            for i, data in enumerate([*dataset]):
                self.state_batch[i] = data[0]
                self.action_batch [i] = data[1] 
                self.next_state_batch[i] = data[2]
                self.reward_batch [i] = data[3]
                self.done_batch [i] = data[4] 

            predict_q_value = self.predict_net(self.state_batch).gather(1, self.action_batch)
            next_state_values = torch.zeros(self.config.Buffer.BATCH_SIZE, device=self.config.Env.device)
            with torch.no_grad():
                next_states = torch.cat([next_state for done, next_state in zip(self.done_batch, self.next_state_batch) if not done]).reshape(-1,self.config.Env.observation_space)
                next_state_values[~self.done_batch] = self.target_net(next_states).max(1)[0]
            
            expected_func = (next_state_values * self.config.Network.GAMMA) + self.reward_batch
            loss = self.criterion(expected_func,predict_q_value.view(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.predict_net.parameters(), 100)
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
        self.predict_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.target_net.load_state_dict(self.predict_net.state_dict())

class DDQNBuffer(DQNBuffer):
    def __init__(self,configs, predict_net, target_net, opt_lossf):
        super().__init__(configs, predict_net, target_net, opt_lossf)
    
    def init_batch(self):
        super().init_batch()

    def add(self,*args):
        super().add(*args)

    def sample(self):
        return super().sample()

    def __len__(self):
        return super().__len__()

    def update(self,*args):
        self.add(*args)

        if len(self.memory) < self.config.Buffer.BATCH_SIZE:
            return
        
        for n in range(self.config.Network.epoch):
            dataset = self.sample()
            for i, data in enumerate([*dataset]):
                self.state_batch[i] = data[0]
                self.action_batch [i] = data[1] 
                self.next_state_batch[i] = data[2]
                self.reward_batch [i] = data[3]
                self.done_batch [i] = data[4] 

            predict_q_value = self.predict_net(self.state_batch).gather(1, self.action_batch)
            next_state_values = torch.zeros(self.config.Buffer.BATCH_SIZE, device=self.config.Env.device)
            with torch.no_grad():
                next_states = torch.cat([next_state for done, next_state in zip(self.done_batch, self.next_state_batch) if not done]).reshape(-1,self.config.Env.observation_space)
                arg_max_a = self.predict_net(next_states).max(1)[1]
                next_state_values[~self.done_batch] = self.target_net(next_states).max(1)[0]
            

            expected_func = (next_state_values * arg_max_a).sum(1) * self.config.Network.GAMMA + self.reward_batch
            loss = self.criterion(expected_func,predict_q_value.view(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.predict_net.parameters(), 100)
            self.optimizer.step()
    
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.predict_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config.Network.TAU + target_net_state_dict[key]*(1- self.config.Network.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def load(self):
        super().save()

    def load(self):
        super().save()

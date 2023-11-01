import random, math, torch, os 
from replay_buffer import ReplayBuffer
from network import QValue 


class DQN():
    def __init__ (self,config):
        self.init_net(config)
        
    def init_net(self,config):
        self.predict_net = QValue(config).to(config.Env.device)
        self.target_net = QValue(config).to(config.Env.device)
        self.target_net.load_state_dict(self.predict_net.state_dict())
        self.opt_lossf = self.init_opt_lossf(config)
        self.init_memory(config)
        self.step_num = 0
        self.config = config
        self.device = self.config.Env.device 

    def init_opt_lossf(self,config):
        optimizer = config.Network.optimizer(self.predict_net.parameters(), lr=config.Network.LR, amsgrad=True)
        criterion = config.Network.loss_function
        return optimizer, criterion

    def init_memory(self,config):
        self.buffer = ReplayBuffer(config, self.predict_net, self.target_net, self.opt_lossf)

    def action(self, state):
        sample = random.random()

        eps_threshold = self.config.Network.EPS_END + (self.config.Network.EPS_START - self.config.Network.EPS_END) * \
            math.exp(-1. * self.step_num / self.config.Network.EPS_DECAY)
        self.step_num += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.predict_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.config.Env.action_space.sample()]], device=self.device, dtype=torch.long)

    def training(self):
        env = self.config.Env.make_env
        for i_episode in range(self.config.Env.num_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            is_done = False
            num_step = 0 

            while num_step <=self.config.Env.max_step or is_done !=True :
                action = self.action(state)

                # state, reward, done, {}
                next_state, reward, done ,_= env.step(action.item()) 
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                self.buffer.update(state, action, next_state, reward, done)
                state = next_state
                
                num_step +=1 
        


                if self.config.Env.training_render:
                    env.render()
                if done:
                    is_done = True
                    break
                
    def testing(self):
        env = self.config.Env.make_env
        self.load()
        for i_episode in range(self.config.Env.te_num_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            is_done = False
            num_step = 0 
            while num_step <=self.config.Env.te_max_step or is_done !=True :
                action = self.action(state)
                # state, reward, done, {}
                next_state, reward, done ,_= env.step(action.item()) 
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                state = next_state

                num_step +=1 

                if done:
                    is_done = True
                    break

                if self.config.Env.testing_render:
                    env.render()

    def save(self):
        path = os.path.join(os.getcwd(),f"{self.config.Env.save_file_name}.pt")
        print(f"save path {path}")
        torch.save(self.target_net.state_dict(), path)
    
    def load(self):
        path = os.path.join(os.getcwd(),f"{self.config.Env.save_file_name}.pt")
        print(f"load path {path}")
        self.target_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.predict_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

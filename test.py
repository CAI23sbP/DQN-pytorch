from dqn import DQN
from configs import Config

if __name__ == "__main__":
    network = DQN(Config)
    print("start_testing")
    network.testing()
